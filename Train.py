# import numpy as np
# import random
# import json

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# from NLTK_Analysis import bag_of_words, tokenize, stem
# from model import NeuralNet

# with open('output_file_999.json', 'r', encoding="utf-8") as json_data:
#     intents = json.load(json_data)

# all_words = []
# tags = []
# xy = []
# # loop through each sentence in our intents patterns
# for intent in intents['intents']:
#     tag = intent['tag']
#     # add to tag list
#     tags.append(tag)
#     for pattern in intent['patterns']:
#         # tokenize each word in the sentence
#         w = tokenize(pattern)
#         # add to our words list
#         all_words.extend(w)
#         # add to xy pair
#         xy.append((w, tag))

# # stem and lower each word
# ignore_words = ['?', '.', '!']
# all_words = [stem(w) for w in all_words if w not in ignore_words]
# # remove duplicates and sort
# all_words = sorted(set(all_words))
# tags = sorted(set(tags))

# print(len(xy), "patterns")
# print(len(tags), "tags:", tags)
# print(len(all_words), "unique stemmed words:", all_words)

# # create training data
# X_train = []
# y_train = []
# for (pattern_sentence, tag) in xy:
#     # X: bag of words for each pattern_sentence
#     bag = bag_of_words(pattern_sentence, all_words)
#     X_train.append(bag)
#     # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
#     label = tags.index(tag)
#     y_train.append(label)

# X_train = np.array(X_train)
# y_train = np.array(y_train)

# # Hyper-parameters 
# num_epochs = 1500
# batch_size = 264
# learning_rate = 0.001
# input_size = len(X_train[0])
# hidden_size = 128
# output_size = len(tags)
# print(input_size, output_size)

# class ChatDataset(Dataset):

#     def __init__(self):
#         self.n_samples = len(X_train)
#         self.x_data = X_train
#         self.y_data = y_train

#     # support indexing such that dataset[i] can be used to get i-th sample
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]

#     # we can call len(dataset) to return the size
#     def __len__(self):
#         return self.n_samples

# dataset = ChatDataset()
# train_loader = DataLoader(dataset=dataset,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           num_workers=0)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# model = NeuralNet(input_size, hidden_size, output_size).to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # ... (الاستيرادات والتحضيرات الأخرى كما هي)

# # Train the model
# for epoch in range(num_epochs):
#     total = 0
#     correct = 0
#     running_loss = 0.0

#     for (words, labels) in train_loader:
#         words = words.to(device)
#         labels = labels.to(dtype=torch.long).to(device)

#         # Forward pass
#         outputs = model(words)
#         loss = criterion(outputs, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * words.size(0)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     # Compute epoch accuracy and loss
#     epoch_loss = running_loss / len(train_loader.dataset)
#     epoch_accuracy = 100 * correct / total

#     if (epoch + 1) % 100 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {epoch_accuracy:.2f}%, Loss: {epoch_loss:.4f}')

# print(f'Final Accuracy: {epoch_accuracy:.2f}%')
# print(f'Final Loss: {epoch_loss:.4f}')

# data = {
#     "model_state": model.state_dict(),
#     "input_size": input_size,
#     "hidden_size": hidden_size,
#     "output_size": output_size,
#     "all_words": all_words,
#     "tags": tags
# }

# FILE = "EVC_model-epoc1500.pth"
# torch.save(data, FILE)

# print(f'Training complete. File saved to {FILE}')









import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from NLTK_Analysis import bag_of_words, tokenize, stem
from model import NeuralNet

with open(r'last\intents.json', 'r', encoding="utf-8") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "EVC_model_slam_V3.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')








# import numpy as np
# import random
# import json

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# from NLTK_Analysis import bag_of_words, tokenize, stem,text_prepare
# from model import NeuralNet,LSTMNet,LSTMModel,GRUModel

# with open('output_file_9.json', 'r', encoding="utf-8") as json_data:
#     training_data = json.load(json_data)

# all_words = []
# tags = []
# xy = []
# for intent in training_data['intents']:
#     tag = intent['tag']
#     tags.append(tag)
#     for response in intent['responses']:
#         text_clen = text_prepare(response)
#         w = tokenize(text_clen)
#         all_words.extend(w)
#         xy.append((w, tag))


# ignore_words = ['?', '.', '!']
# all_words = [stem(w) for w in all_words if w not in ignore_words]
# all_words = sorted(set(all_words))
# tags = sorted(set(tags))

# print(len(xy), "responses")
# print(len(tags), "tags:", tags)
# print(len(all_words), "unique stemmed words:", all_words)

# X_train = []
# y_train = []
# for (pattern_sentence, tag) in xy:
#     bag = bag_of_words(pattern_sentence, all_words)
#     X_train.append(bag)
#     label = tags.index(tag)
#     y_train.append(label)

# X_train = np.array(X_train)
# y_train = np.array(y_train)

# num_epochs = 500
# batch_size = 32
# learning_rate = 0.001
# input_size = len(X_train[0])
# hidden_size = 64
# output_size = len(tags)
# print(input_size, output_size)

# class ChatDataset(Dataset):

#     def __init__(self):
#         self.n_samples = len(X_train)
#         self.x_data = X_train
#         self.y_data = y_train

#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]

#     def __len__(self):
#         return self.n_samples

# dataset = ChatDataset()
# train_loader = DataLoader(dataset=dataset,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           num_workers=0)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = NeuralNet(input_size, hidden_size, output_size).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Train the model
# for epoch in range(num_epochs):
#     total = 0
#     correct = 0
#     running_loss = 0.0

#     for (words, labels) in train_loader:
#         words = words.to(device)
#         labels = labels.to(dtype=torch.long).to(device)

#         # Forward pass
#         outputs = model(words)
#         loss = criterion(outputs, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * words.size(0)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     # Compute epoch accuracy and loss
#     epoch_loss = running_loss / len(train_loader.dataset)
#     epoch_accuracy = 100 * correct / total

#     if (epoch + 1) % 100 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {epoch_accuracy:.2f}%, Loss: {epoch_loss:.4f}')

# print(f'Final Accuracy: {epoch_accuracy:.2f}%')
# print(f'Final Loss: {epoch_loss:.4f}')

# data = {
#     "model_state": model.state_dict(),
#     "input_size": input_size,
#     "hidden_size": hidden_size,
#     "output_size": output_size,
#     "all_words": all_words,
#     "tags": tags
# }

# FILE = "data-v2.pth"
# torch.save(data, FILE)

# print(f'Training complete. File saved to {FILE}')







# import numpy as np
# import random
# import json

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# from NLTK_Analysis import bag_of_words, tokenize, stem
# from model import NeuralNet
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Traning on: ",device)
# with open('output_file_9.json', 'r', encoding="utf-8") as json_data:
#     intents = json.load(json_data)

# all_words = []
# tags = []
# xy = []
# # loop through each sentence in our intents patterns
# for intent in intents['intents']:
#     tag = intent['tag']
#     # add to tag list
#     tags.append(tag)
#     for pattern in intent['patterns']:
#         # tokenize each word in the sentence
#         w = tokenize(pattern)
#         # add to our words list
#         all_words.extend(w)
#         # add to xy pair
#         xy.append((w, tag))

# # stem and lower each word
# ignore_words = ['?', '.', '!']
# all_words = [stem(w) for w in all_words if w not in ignore_words]
# # remove duplicates and sort
# all_words = sorted(set(all_words))
# tags = sorted(set(tags))

# print(len(xy), "patterns")
# print(len(tags), "tags:", tags)
# print(len(all_words), "unique stemmed words:", all_words)

# # create training data
# X_train = []
# y_train = []
# for (pattern_sentence, tag) in xy:
#     # X: bag of words for each pattern_sentence
#     bag = bag_of_words(pattern_sentence, all_words)
#     X_train.append(bag)
#     # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
#     label = tags.index(tag)
#     y_train.append(label)

# X_train = np.array(X_train)
# y_train = np.array(y_train)

# # Hyper-parameters 
# num_epochs = 500
# batch_size = 264
# learning_rate = 0.001
# input_size = len(X_train[0])
# hidden_size = 64
# output_size = len(tags)
# print(input_size, output_size)

# class ChatDataset(Dataset):

#     def __init__(self):
#         self.n_samples = len(X_train)
#         self.x_data = X_train
#         self.y_data = y_train

#     # support indexing such that dataset[i] can be used to get i-th sample
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]

#     # we can call len(dataset) to return the size
#     def __len__(self):
#         return self.n_samples

# dataset = ChatDataset()
# train_loader = DataLoader(dataset=dataset,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           num_workers=0)


# model = NeuralNet(input_size, hidden_size, output_size).to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Train the model

# for epoch in range(num_epochs):
#     total = 0
#     correct = 0
#     running_loss = 0.0

#     for (words, labels) in train_loader:
#         words = words.to(device)
#         labels = labels.to(dtype=torch.long).to(device)

#         # Forward pass
#         outputs = model(words)
#         loss = criterion(outputs, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * words.size(0)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     # Compute epoch accuracy and loss
#     epoch_loss = running_loss / len(train_loader.dataset)
#     epoch_accuracy = 100 * correct / total

#     if (epoch + 1) % 100 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {epoch_accuracy:.2f}%, Loss: {epoch_loss:.4f}')

# print(f'Final Accuracy: {epoch_accuracy:.2f}%')
# print(f'Final Loss: {epoch_loss:.4f}')



# data = {
# "model_state": model.state_dict(),
# "input_size": input_size,
# "hidden_size": hidden_size,
# "output_size": output_size,
# "all_words": all_words,
# "tags": tags
# }

# FILE = "data-v4.pth"
# torch.save(data, FILE)

# print(f'training complete. file saved to {FILE}')







