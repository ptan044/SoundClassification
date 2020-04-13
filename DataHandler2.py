from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
import soundfile
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.DEBUG)

def _stft(
    waveform,
    n_fft,
    hop_length,
    win_length,
    window,
    center,
    pad_mode,
    normalized,
    onesided,
):
    # type: (Tensor, int, Optional[int], Optional[int], Optional[Tensor], bool, str, bool, bool) -> Tensor
    return torch.stft(
        waveform,
        n_fft,
        hop_length,
        win_length,
        window,
        center,
        pad_mode,
        normalized,
        onesided,
    )

def complex_norm(complex_tensor, power=1.0):
    # type: (Tensor, float) -> Tensor
    r"""Compute the norm of complex tensor input.
    Args:
        complex_tensor (torch.Tensor): Tensor shape of `(..., complex=2)`
        power (float): Power of the norm. (Default: `1.0`).
    Returns:
        torch.Tensor: Power of the normed input tensor. Shape of `(..., )`
    """
    if power == 1.0:
        return torch.norm(complex_tensor, 2, -1)
    return torch.norm(complex_tensor, 2, -1).pow(power)

def spectrogram(
    waveform, pad, window, n_fft, hop_length, win_length, power, normalized
):
    # type: (Tensor, int, Tensor, int, int, int, Optional[float], bool) -> Tensor
    r"""Create a spectrogram or a batch of spectrograms from a raw audio signal.
    The spectrogram can be either magnitude-only or complex.
    Args:
        waveform (torch.Tensor): Tensor of audio of dimension (..., time)
        pad (int): Two sided padding of signal
        window (torch.Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT
        hop_length (int): Length of hop between STFT windows
        win_length (int): Window size
        power (float): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
            If None, then the complex spectrum is returned instead.
        normalized (bool): Whether to normalize by magnitude after stft
    Returns:
        torch.Tensor: Dimension (..., freq, time), freq is
        ``n_fft // 2 + 1`` and ``n_fft`` is the number of
        Fourier bins, and time is the number of window hops (n_frame).
    """

    if pad > 0:
        # TODO add "with torch.no_grad():" back when JIT supports it
        waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

    # pack batch
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

    # default values are consistent with librosa.core.spectrum._spectrogram
    spec_f = _stft(
        waveform=waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True,
        pad_mode="reflect", normalized=False, onesided=True
        )

    # unpack batch
    spec_f = spec_f.view(shape[:-1] + spec_f.shape[-3:])

    if normalized:
        spec_f /= window.pow(2.).sum().sqrt()
    if power is not None:
        spec_f = complex_norm(spec_f, power=power)

    return spec_f


class Spectrogram(torch.nn.Module):
    r"""Create a spectrogram from a audio signal
    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins
        win_length (int): Window size. (Default: ``n_fft``)
        hop_length (int, optional): Length of hop between STFT windows. (
            Default: ``win_length // 2``)
        pad (int): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[[...], torch.Tensor]): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
        normalized (bool): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (Dict[..., ...]): Arguments for window function. (Default: ``None``)
    """
    __constants__ = ['n_fft', 'win_length', 'hop_length', 'pad', 'power', 'normalized']

    def __init__(self, n_fft=400, win_length=None, hop_length=None,
                 pad=0, window_fn=torch.hann_window,
                 power=2., normalized=False, wkwargs=None):
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.pad = pad
        self.power = power
        self.normalized = normalized

    def forward(self, waveform):
        r"""
        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (..., time)
        Returns:
            torch.Tensor: Dimension (..., freq, time), where freq is
            ``n_fft // 2 + 1`` where ``n_fft`` is the number of
            Fourier bins, and time is the number of window hops (n_frame).
        """
        return spectrogram(waveform, self.pad, self.window, self.n_fft, self.hop_length,
                             self.win_length, self.power, self.normalized)
class subDataSet(Dataset):
    """
    This class is used to segment each wav file into clips
    After STFT the number of time frames in each clip is 128
    STFT is NOT applied within this class
    """
    def __init__(self, wav_filename, meta_filename, crop_duration_s, transform=None):
        self.transform = transform
        self.eventList = ['clearthroat', 'cough', 'doorslam', 'drawer', 'keyboard', 'keysDrop', 'knock', 'laughter',
                          'pageturn', 'phone', 'speech']

        self.__wav_filename = wav_filename
        self.__meta_filename = meta_filename
        self.__desired_crop_duration_s = crop_duration_s


        waveform, sample_rate = soundfile.read(
            self.__wav_filename, dtype="float32", always_2d=True
        )
        waveform = torch.from_numpy(waveform).t()

        self.__TD_data_size = int(self.__desired_crop_duration_s*sample_rate)
        self.__len = waveform.size()[1]//self.__TD_data_size

    def __getitem__(self, idx):
        """
        This method gets each wav file as well as the type of events and when they occur in the reccording
        Parameters
        ----------
        idx: Index of which wav file selected

        Returns
        -------

        """
        start_frame = (idx)*self.__TD_data_size
        next_start_frame = (idx+1)*self.__TD_data_size
        end_frame = next_start_frame - 1
        channel = 0

        waveform, sample_rate = soundfile.read(
            self.__wav_filename, dtype="float32", always_2d=True
        )
        waveform = torch.from_numpy(waveform).t()


        meta_frame = pd.read_csv(self.__meta_filename)
        events = meta_frame.iloc[0:, 0]
        start_times = meta_frame.iloc[0:, 1].astype(float)
        end_times = meta_frame.iloc[0:, 2].astype(float)

        waveform = waveform[channel:channel+1,start_frame:next_start_frame]
        event_prob = torch.zeros(11)

        for eventIndex in range(len(events)):
            eventName = events[eventIndex]
            event_start = start_times[eventIndex]*48000
            event_end = end_times[eventIndex]*48000
            if start_frame <= event_start <= end_frame:

                prob = float((min(end_frame,event_end) - event_start)/self.__TD_data_size)
                eventValue = self.eventList.index(eventName)
                event_prob[eventValue] =float(event_prob[eventValue]+prob)
                pass




        sample = {'waveform': waveform, 'sample_rate': sample_rate, 'event_prob': event_prob}

        if self.transform:
            sample = self.transform(sample)

        return sample
        # return waveform,sample_rate,events,start_times,end_times

    def __len__(self):
        return self.__len


class Spectrogram1(object):
    def __init__(self, dummy=None, n_fft=2048, hop_length=960, pad=0, window_fn=torch.hann_window,power=2):
        self.__n_fft=n_fft
        self.__hop_length = hop_length
        self.__pad = pad
        self.__window_fn = window_fn
        self.__power = power


    def __call__(self, sample):

        stft = Spectrogram(n_fft=self.__n_fft, hop_length=self.__hop_length,
                                                 pad=self.__pad, window_fn=self.__window_fn,
                                                 power=self.__power)


        specgram = stft(sample['waveform'])
        specData = {'specgram': specgram, 'sample_rate': sample['sample_rate'],'event_prob': sample['event_prob']}


        return(specData)

class Binarize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        event_prob = sample['event_prob']
        new_prob = torch.zeros(11,dtype=torch.int)
        for i in range(len(event_prob)):
            if event_prob[i] ==0:
                pass
            else:
                new_prob[i] =1


        sample['event_prob'] = new_prob


        return(sample)



if __name__ == '__main__':
    wav_dir = "C:\\Users\\jgohj\\PycharmProjects\\Jon\\data\\mic_dev_test"
    meta_dir = "C:\\Users\\jgohj\\PycharmProjects\\Jon\\data\\metadata_dev"
    FRAMES_AFTER_STFT = 128
    STFT_HOP_SIZE_S = 0.02

    crop_duration_s = (FRAMES_AFTER_STFT - 1) * STFT_HOP_SIZE_S

    specific_name = "split1_ir0_ov1_4"
    wav_filename = os.path.join(wav_dir,specific_name+".wav")
    meta_filename = os.path.join(meta_dir,specific_name+".csv")
    print("loading filename:{}".format(wav_filename))
    print("loading meta_data:{}".format(meta_dir))




    #region subDataSet test
    testingSet = subDataSet(wav_filename, meta_filename, crop_duration_s=crop_duration_s)
    bob = testingSet[0]
    print(bob['waveform'])
    print("the event probs are:\n {}".format(bob['event_prob']))

    # endregion

    # region subDataSet Spectrogram Trasnform Test
    composed = transforms.Compose([Spectrogram1()])
    specDataSet = subDataSet(wav_filename, meta_filename, crop_duration_s=crop_duration_s, transform=composed)
    specbob = specDataSet[1]
    print(specbob['specgram'].size())
    # endregion

    # region subDataSet concatenate test
    wav_dir = "C:\\Users\\jgohj\\PycharmProjects\\Jon\\data\\mic_dev_test"
    meta_dir = "C:\\Users\\jgohj\\PycharmProjects\\Jon\\data\\metadata_dev"

    datasets = []

    specific_name = "split1_ir0_ov1_4"
    wav_filename = os.path.join(wav_dir, specific_name + ".wav")
    meta_filename = os.path.join(meta_dir, specific_name + ".csv")
    set1 = subDataSet(wav_filename, meta_filename, crop_duration_s=crop_duration_s, transform=composed)

    specific_name = "split1_ir0_ov1_2"
    wav_filename = os.path.join(wav_dir, specific_name + ".wav")
    meta_filename = os.path.join(meta_dir, specific_name + ".csv")
    set2 = subDataSet(wav_filename, meta_filename, crop_duration_s=crop_duration_s, transform=composed)


    datasets.append(set1)
    datasets.append(set2)
    dataset = ConcatDataset(datasets)
    print(dataset[0]['specgram'].size())
    print(dataset[23]['specgram'].size())
    print(os.listdir(wav_dir).__len__())
    # end region

    # region fulltest
    wav_dir = "C:\\Users\\jgohj\\PycharmProjects\\Jon\\data\\mic_dev_test"
    meta_dir = "C:\\Users\\jgohj\\PycharmProjects\\Jon\\data\\metadata_dev"
    wav_list = os.listdir((wav_dir))
    composed = transforms.Compose([Spectrogram1(),Binarize()])
    melcomposed = transforms.Compose([MelSpectrogram(), Binarize()])
    datasets = []
    meldatasets = []

    print("Creating Full Data set...")
    for wav_filename in wav_list:
        specific_name = wav_filename.split(".wav",1)[0]
        wav_filename = os.path.join(wav_dir, specific_name + ".wav")
        meta_filename = os.path.join(meta_dir, specific_name + ".csv")
        set = subDataSet(wav_filename,meta_filename,crop_duration_s=crop_duration_s,transform=composed)
        datasets.append(set)
        melset = subDataSet(wav_filename, meta_filename, crop_duration_s=crop_duration_s, transform=melcomposed)
        meldatasets.append(melset)
    dataset = ConcatDataset(datasets)
    meldataset = ConcatDataset(meldatasets)
    print("Full Data set created")
    print(dataset[2]['event_prob'])

