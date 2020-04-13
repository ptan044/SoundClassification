import torch
from torch.utils.data import ConcatDataset
from torchvision import transforms
import os
from SoundEventClassification.DataHandler2 import subDataSet, Spectrogram1, Binarize

eventList = ['clearthroat', 'cough', 'doorslam', 'drawer', 'keyboard', 'keysDrop', 'knock', 'laughter',
                          'pageturn', 'phone', 'speech']

def classifier(prediction):
    classifiedPrediction = torch.zeros_like(prediction, dtype=torch.int)
    for index in range(len(prediction)):
        if prediction[index] < 0.5:
            classifiedPrediction[index] = 0
        else:
            classifiedPrediction[index] = 1

    return classifiedPrediction

# torch.cuda.set_device(7)
crop_duration = int(0.02 * 127)


model_name = "./2020-01-16_1/network_11"

device = torch.device("cpu")

net = torch.load(model_name,map_location=torch.device('cpu'))
# net.to(device)
# region Loading Data
composed = transforms.Compose([Spectrogram1(), Binarize()])

wav_validation_dir = "data/mic_dev_val"
# wav_dir = "data/mic_dev_train"
meta_dir = "data/metadata_dev"
wav_validation_list = os.listdir((wav_validation_dir))
# wav_list = os.listdir((wav_dir))
datasets = []

print("creating validation dataset....")

for wav_filename in wav_validation_list:
    specific_name = wav_filename.split(".wav", 1)[0]
    wav_filename = os.path.join(wav_validation_dir, specific_name + ".wav")
    meta_filename = os.path.join(meta_dir, specific_name + ".csv")
    set = subDataSet(wav_filename, meta_filename, crop_duration_s=crop_duration, transform=composed)
    datasets.append(set)
validationset = ConcatDataset(datasets)
print("validation dataset created")

results = torch.zeros(11, 6)
for data in validationset:
    #labels = data['event_prob']
    inputs = data['specgram'][None, :, :]
    #print(inputs.size())
    inputs = inputs.to(device)
    print(inputs.shape)

    #labels = labels.to(device)
    output = net(inputs)
    predicted_prob = torch.sigmoid(output) #formula to predit prob
    predicted_events = classifier(predicted_prob[0])
    #print(predicted_prob.tolist()) # .tolist() - convert tensor to list
    #print(predicted_events) # reference for showing the identified binary class of sound.

    # eventList = ['clearthroat(0)', 'cough(1)', 'doorslam(2)', 'drawer(3)', 'keyboard(4)', 'keysDrop(5)', 'knock(6)', 'laughter(7)',
    #              'pageturn(8)', 'phone(9)', 'speech(10)']   #for reference


    prob = 0 #initialise prob =0
    for index in range(len(predicted_events)):

        if predicted_events[index] == 1:
            if float(predicted_prob[0,index].tolist()) >= prob: #float for decimal figures. [0,index].tolist() - convert tensor to list and select element from the list
                prob = predicted_prob[0,index].tolist() #if the latter prob is higher, replace prob and sound class
                sound = eventList[index]
    if not prob == 0: # if prob /= 0, print class of sound and prob. else the latter.
        print("The sound classification identify is: {}".format(sound))
        print("The predicted probabilities is: {}".format(prob))
    else:
        print("Unidentified class of sound!!!")


#     for index in range(len(labels)):
#         if labels[index] == predicted_events[index]:
#             results[index, 0] = results[index, 0] + 1
#         else:
#             results[index, 1] = results[index, 1] + 1
#         if labels[index] == 1 and predicted_events[index] == 1:
#             results[index, 2] = results[index, 2] + 1
#         elif labels[index] == 0 and predicted_events[index] == 0:
#             results[index, 3] = results[index, 3] + 1
#         elif labels[index] == 0 and predicted_events[index] == 1:
#             results[index, 4] = results[index, 4] + 1
#         elif labels[index] == 1 and predicted_events[index] == 0:
#             results[index, 5] = results[index, 5] + 1
#
# evaluation = {"accuracy": [], "precision": [], "recall": [], "f1": []}
#
# for event_results in results:
#     accuracy = (event_results[2] + event_results[3]) / (
#                 event_results[2] + event_results[3] + event_results[4] + event_results[5])
#     precision = event_results[2] / (event_results[2] + event_results[4])
#     recall = event_results[2] / (event_results[2] + event_results[5])
#     f1 = 2 * (recall * precision) / (recall + precision)
#     evaluation["accuracy"].append(accuracy.item())
#     evaluation["precision"].append(precision.item())
#     evaluation["recall"].append(recall.item())
#     evaluation["f1"].append(f1.item())
#
# print(evaluation)