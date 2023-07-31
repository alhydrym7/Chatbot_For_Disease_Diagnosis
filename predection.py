# import random
# import json

# import torch

# from model import NeuralNet,LSTMModel
# from NLTK_Analysis import bag_of_words, tokenize

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# with open(r'C:\Users\asus\Desktop\Bayan\output_file_10.json', 'r', encoding="utf-8") as json_data:
#     intents = json.load(json_data)

# FILE = "data-v1.pth"
# data = torch.load(FILE)

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data['all_words']
# tags = data['tags']
# model_state = data["model_state"]

# model = LSTMModel(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()

# bot_name = "Sam"
# print("Let's chat! (type 'quit' to exit)")
# while True:
#     # sentence = "do you use credit cards?"
#     sentence = input("You: ")
#     if sentence == "quit":
#         break

#     sentence = tokenize(sentence)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)

#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     print(prob.item())
#     if prob.item() > 0.20:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 print(f"{bot_name}: {random.choice(intent['responses'])}")
#     else:
#         print(f"{bot_name}: I do not understand...")




# import torch
# import random
# import json


# with open('output_file_9.json', 'r', encoding="utf-8") as json_data:
#     intents = json.load(json_data)

# # Load the saved model information
# FILE = "data.pth"
# data = torch.load(FILE)

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data["all_words"]
# tags = data["tags"]
# model_state = data["model_state"]

# # Load the model architecture
# from model import NeuralNet
# model = NeuralNet(input_size, hidden_size, output_size)
# model.load_state_dict(model_state)
# model.eval()

# bot_name = "EVC: "

# import nltk
# from NLTK_Analysis import bag_of_words, tokenize, stem

# def predict_intent(sentence):
#     # Tokenize and stem the input sentence
#     sentence = tokenize(sentence)
#     sentence = [stem(word) for word in sentence]

#     # Create a bag of words representation for the sentence
#     X = bag_of_words(sentence, all_words)

#     # Convert to a PyTorch tensor and add batch dimension
#     X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

#     # Make the prediction using the trained model
#     output = model(X)

#     # Get the index of the highest value (the predicted class)
#     _, predicted_idx = torch.max(output, dim=1)
#         # Get the predicted tag from the index
#     predicted_tag = tags[predicted_idx.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted_idx.item()]
#     print(prob.item())
#     if prob.item() > 0:
#         for intent in intents['intents']:
#             if predicted_tag == intent["tag"]:
#                 print(f"{bot_name}: {intent['responses']}")
#     else:
#         print(f"{bot_name}: I do not understand...")



#     return predicted_tag

# t= "Neuropathies: The Nerve Damage of Diabete"
# input_sentence = 'How many people are affected by pyruvate carboxylase deficiency'

# predicted_tag = predict_intent(input_sentence)
# print(f"Predicted tag for '{input_sentence}': {predicted_tag}")











# import random
# import json

# import torch

# from model import NeuralNet
# from NLTK_Analysis import bag_of_words, tokenize, stem

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# # Load the data for N-grams and vocabulary
# FILE = "data.pth"
# data = torch.load(FILE)

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data['all_words']
# tags = data['tags']
# model_state = data["model_state"]

# # Convert all_words to N-grams
# ngrams = 2  # ثنائيات (Bigrams)
# all_words_ngrams = []
# for w in all_words:
#     w = w.split()  # تحويل الثنائيات إلى قائمة الكلمات المكونة
#     for i in range(len(w) - ngrams + 1):
#         ngram = w[i:i + ngrams]
#         all_words_ngrams.append(' '.join(ngram))
# all_words_ngrams = sorted(set(all_words_ngrams))  # إزالة الكلمات المكررة والترتيب

# # نقوم بنفس العملية لجمل الاستفهام (sentences)
# for i in range(len(intents['intents'])):
#     patterns = intents['intents'][i]['patterns']
#     for j in range(len(patterns)):
#         sentence = tokenize(patterns[j])
#         for k in range(len(sentence) - ngrams + 1):
#             ngram = sentence[k:k + ngrams]
#             sentence[k] = ' '.join(ngram)
#         intents['intents'][i]['patterns'][j] = ' '.join(sentence)

# # إعادة بناء مصفوفة all_words الخاصة بالنموذج لتشمل الثنائيات أيضًا
# all_words = sorted(set(all_words + all_words_ngrams))

# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()

# bot_name = "Sam"
# print("Let's chat! (type 'quit' to exit)")
# while True:
#     sentence = input("You: ")
#     if sentence == "quit":
#         break

#     sentence = tokenize(sentence)
#     for i in range(len(sentence) - ngrams + 1):
#         ngram = sentence[i:i + ngrams]
#         sentence[i] = ' '.join(ngram)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)

#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 print(f"{bot_name}: {random.choice(intent['responses'])}")
#     else:
#         print(f"{bot_name}: I do not understand...")












# import random
# import json

# import torch

# from model import NeuralNet
# from NLTK_Analysis import bag_of_words, tokenize

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# with open('output_file_999.json', 'r', encoding="utf-8") as json_data:
#     intents = json.load(json_data)

# FILE = "EVC_model.pth"
# data = torch.load(FILE, map_location=torch.device('cpu'))

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data['all_words']
# tags = data['tags']
# model_state = data["model_state"]

# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()

# bot_name = "Sam"
# print("Let's chat! (type 'quit' to exit)")
# while True:
#     # sentence = "do you use credit cards?"
#     sentence = input("You: ")
#     if sentence == "quit":
#         break

#     sentence = tokenize(sentence)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)

#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 prd = intent['responses'][0]
#                 print(prd)
#     else:
#         print(f"{bot_name}: I do not understand...")





import random
import json

import torch

from model import NeuralNet
from NLTK_Analysis import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(r'last\intents.json', 'r', encoding="utf-8") as json_data:
    intents = json.load(json_data)

FILE = 'EVC_model_slam.pth'
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(prob)
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")












