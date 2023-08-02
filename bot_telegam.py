


#================================================================





from model import NeuralNet
from NLTK_Analysis import bag_of_words, tokenize
import telebot
import json
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import difflib
from datetime import datetime
import threading
import time




current_datetime = datetime.now()



BOT_TOKEN = "6505003602:AAFsZnZgAb9WdmJu7O4Djxoa0cmF1_r2YbA"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ann_model = tf.keras.models.load_model(r"C:\Users\asus\Downloads\ann_model.h5")


with open(r'last\intents.json', 'r', encoding="utf-8") as json_data:
    intents_ = json.load(json_data)

patterns_l = [pattern.lower() for intent in intents_['intents'] for pattern in intent['patterns']]
patterns_c = [pattern for intent in intents_['intents'] for pattern in intent['patterns']]




from googletrans import Translator

def translate_to_arabic(text):
    result = [None]  # A mutable list to store the result from the thread

    def translation_thread():
        nonlocal result
        translator = Translator()
        translated_text = translator.translate(text, src='en', dest='ar')
        result[0] = translated_text.text

    thread = threading.Thread(target=translation_thread)
    thread.start()

    # Wait for the thread to complete, but with a timeout
    thread.join(timeout=10)

    if thread.is_alive():
        thread.join()  

    translated_text = result[0]
    return translated_text
def translate_to_english(text):
    result = [None]  # A mutable list to store the result from the thread

    def translation_thread():
        nonlocal result
        translator = Translator()
        translated_text = translator.translate(text, src='ar', dest='en')
        result[0] = translated_text.text

    thread = threading.Thread(target=translation_thread)
    thread.start()

    # Wait for the thread to complete, but with a timeout
    thread.join(timeout=10)

    # Check if the thread is still running (i.e., translation took more than 10 seconds)
    if thread.is_alive():
        thread.join()  # Forcefully join the thread if it's still running

    # Retrieve the translation result from the result list
    translated_text = result[0]
    return translated_text





def leadMyWord(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = word_tokenize(text, language='english')
    text = [word for word in text if not word in stopwordSet]
    text = " ".join(text)
    return text

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = word_tokenize(text, language='english')
    text = [word for word in text if not word in stopwordSet]
    text = " ".join(text)
    return text

def ann_predict_with_confidence(text):
    puretext = leadMyWord(text)
    vector = tfidv.transform([puretext])
    vector.sort_indices()
    predicted = ann_model.predict(vector)
    predicted_category = predicted.argmax(axis=1)
    confidence = predicted.max()
    predicted_class = le.classes_[predicted_category][0]

    return predicted_class, confidence

def autocorrect(input_sentence, stored_sentences):
    if input_sentence is None:
        return "Sorry, I couldn't understand your input."
    
    similarity_scores = []
    for stored_sentence in stored_sentences:
        similarity_scores.append(difflib.SequenceMatcher(None, input_sentence, stored_sentence).ratio())
    max_score = max(similarity_scores)
    if max_score >= 0.5:
        index = similarity_scores.index(max_score)
        return stored_sentences[index]
    else:
        return "Sorry, I couldn't find a match for your sentence."
def model_slam(text):
        
        import random
        import json

        import torch

        from model import NeuralNet
        from NLTK_Analysis import bag_of_words, tokenize

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(r'last\intents.json', 'r', encoding="utf-8") as json_data:
            intents = json.load(json_data)

        FILE = 'EVC_model_slam_V3.pth'
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

        # sentence = "do you use credit cards?"
        sentence = text

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
        if prob.item() > 0.5:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    
                    if tag == "date":
                        responce = random.choice(intent['responses'])
                        formatted_datetime = current_datetime.strftime("%Y-%m-%d")
                        prd = responce +f" {formatted_datetime}"
                        return prd
                    else:
                        if prob.item() > 0.75:
                            responce = random.choice(intent['responses'])
                            return responce
        else:
            print(f"{bot_name}: I do not understand...")

def more_iform(predicted_class):
            with open('output_file_999.json', 'r', encoding="utf-8") as json_data:
                intents = json.load(json_data)

            FILE = "EVC_model.pth"
            data = torch.load(FILE, map_location=torch.device('cpu'))

            input_size = data["input_size"]
            hidden_size = data["hidden_size"]
            output_size = data["output_size"]
            all_words = data['all_words']
            tags = data['tags']
            model_state = data["model_state"]

            model = NeuralNet(input_size, hidden_size, output_size).to(device)
            model.load_state_dict(model_state)
            model.eval()
            sentence = predicted_class
            sentence = tokenize(sentence)
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)

            output = model(X)
            _, predicted = torch.max(output, dim=1)

            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            if prob.item() > 0.75:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        prd = intent['responses']
                        return prd


stopwordSet = set(stopwords.words('english'))

df = pd.read_csv(r'C:\Users\asus\Desktop\Artificial intelligence\Third Year\Summer\EVC\Dataanlysez\project\Symptom2Disease.csv')
data = df[['label','text']]
le = LabelEncoder()
labelEncode = le.fit_transform(data["label"])
categorical_y = to_categorical(labelEncode)


tfidv = TfidfVectorizer(max_features=20001)
textList = data.text.apply(leadMyWord)
textList = list(textList)
x = tfidv.fit_transform(textList)
x.sort_indices()

bot_name = "Drons Team: "
bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start'])
def start(update, context):
    update.message.reply_text("مرحباً بك ، مالذي تشعر به؟")


def handle_message(update, context):
    user_input = update.message.text
    user_input = translate_to_english(user_input)

    # Check if model_slam response is not empty
    if user_input == 'tell me about yourself':
        user_input = 'can you tell me about yourself'
    patterns_l2 = autocorrect(user_input, patterns_l)
    patterns_c2 = autocorrect(user_input, patterns_c)

    if (patterns_l2 in patterns_l) or (patterns_c2 in patterns_c):
        print('Pattern')
        model_slam_response = model_slam(user_input)
        if model_slam_response:
            responce_ar = translate_to_arabic(model_slam_response)
            update.message.reply_text(responce_ar)
        else:
            update.message.reply_text("عذرًا ، لا أستطع معالجة طلبك.")
    else:
        # Check if the user previously responded to more_info_prompt
        if "waiting_for_response" in context.user_data:
            handle_more_info_response(update, context)  # Handle the response
        else:
            # Use the ANN model to predict the disease category and confidence
            predicted_class, confidence = ann_predict_with_confidence(user_input)
            response = f"You have been diagnosed with a disease: {predicted_class}"
            responce_ar = translate_to_arabic(response)

            if confidence >= 0.10:
                more_info_prompt = "هل تريد المزيد من التفاصيل حول المرض المتوقع؟ (نعم / لا): "
                update.message.reply_text(responce_ar)
                update.message.reply_text(more_info_prompt)
                # Set a state to wait for the user's response
                context.user_data["waiting_for_response"] = True
                # Set the predicted_class to the context for later use
                context.user_data["predicted_class"] = predicted_class
            if confidence >= 0.5 and confidence < 0.10:
                update.message.reply_text("هل يمكنك تقديم مزيد من المعلومات حول الأعراض التي تعاني منها؟")


            if  confidence < 0.5:
                pass
                # context.user_data.pop("waiting_for_response", None)  # Clear waiting state if any
                # update.message.reply_text("انا اسف حقا لم استطيع ان افهم ماذا تريد")



def handle_more_info_response(update, context):
    user_response = update.message.text.strip().lower()
    user_response =translate_to_english(user_response).lower()
    print(user_response)
    

    # Check if the user responded with "yes" or "no"
    if user_response == "yes":
        # Retrieve the predicted_class from the context
        predicted_class = context.user_data.get("predicted_class")
        if predicted_class:
            more_info = more_iform(predicted_class)
            # responce_ar = translate_to_arabic(more_info)

            if more_info == "null":
                ex = "انا اعتذر منك لااستطيع ان اقدم لك المزيد من المعلومات عن هذا المرض"
                update.message.reply_text(ex)
            else:
                update.message.reply_text(more_info)
        else:
            update.message.reply_text("أنا آسف ، لم أتمكن من العثور على الفصل المتوقع.")
    elif user_response == "no":
        update.message.reply_text("حسنًا ، إذا كان لديك أي أسئلة أخرى ، فلا تتردد في طرحها.")
        # Optionally, you can reset the waiting state for the next interaction
        context.user_data.pop("waiting_for_response", None)
    else:
        # If the user provided an invalid response, prompt them again
        update.message.reply_text("الرجاء الرد ' بنعم'  أو ' لا'.")

    # Clear the waiting state after processing the response
    context.user_data.pop("waiting_for_response", None)

# Main function to run the bot
def main():
    updater = Updater(token=BOT_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Handlers for commands and messages
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    dispatcher.add_handler(MessageHandler(Filters.regex(r'^(yes|no)$'), handle_more_info_response))

    # Start the bot
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()




















# from model import NeuralNet
# from NLTK_Analysis import bag_of_words, tokenize
# import telebot
# import json
# import re
# from nltk.corpus import stopwords
# from nltk import word_tokenize
# import pandas as pd
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# import torch
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder
# from telegram.ext import Updater, CommandHandler, MessageHandler, Filters



# BOT_TOKEN = "6505003602:AAFsZnZgAb9WdmJu7O4Djxoa0cmF1_r2YbA"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ann_model = tf.keras.models.load_model(r"C:\Users\asus\Downloads\ann_model.h5")
# # كود النموذج الثاني
# def leadMyWord(text):
#     text = re.sub('[^a-zA-Z]', ' ', text)
#     text = text.lower()
#     text = word_tokenize(text, language='english')
#     text = [word for word in text if not word in stopwordSet]
#     text = " ".join(text)
#     return text

# def preprocess_text(text):
#     text = re.sub('[^a-zA-Z]', ' ', text)
#     text = text.lower()
#     text = word_tokenize(text, language='english')
#     text = [word for word in text if not word in stopwordSet]
#     text = " ".join(text)
#     return text

# def ann_predict_with_confidence(text):
#     puretext = leadMyWord(text)
#     vector = tfidv.transform([puretext])
#     vector.sort_indices()
#     predicted = ann_model.predict(vector)
#     predicted_category = predicted.argmax(axis=1)
#     confidence = predicted.max()
#     predicted_class = le.classes_[predicted_category][0]

#     return predicted_class, confidence

# # كود النموذج الثالث
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
# def more_iform(predicted_class):
        
#             # استخدام النموذج الثالث للحصول على مزيد من التفاصيل حول المرض
#             sentence = predicted_class
#             sentence = tokenize(sentence)
#             X = bag_of_words(sentence, all_words)
#             X = X.reshape(1, X.shape[0])
#             X = torch.from_numpy(X).to(device)

#             output = model(X)
#             _, predicted = torch.max(output, dim=1)

#             tag = tags[predicted.item()]

#             probs = torch.softmax(output, dim=1)
#             prob = probs[0][predicted.item()]
#             if prob.item() > 0.75:
#                 for intent in intents['intents']:
#                     if tag == intent["tag"]:
#                         prd = intent['responses']
#                         return prd


# # استخدام الكود الأول للتفاعل مع المستخدم
# stopwordSet = set(stopwords.words('english'))

#     # تحميل وتجهيز الملف الذي يحتوي على البيانات
# df = pd.read_csv(r'C:\Users\asus\Desktop\Artificial intelligence\Third Year\Summer\EVC\Dataanlysez\project\Symptom2Disease.csv')
# data = df[['label','text']]
# le = LabelEncoder()
# labelEncode = le.fit_transform(data["label"])
# categorical_y = to_categorical(labelEncode)


# # تنفيذ نفس الخطوات التي تم تنفيذها سابقًا لتحديد متغير tfidv
# tfidv = TfidfVectorizer(max_features=20001)
# textList = data.text.apply(leadMyWord)
# textList = list(textList)
# x = tfidv.fit_transform(textList)
# x.sort_indices()

# bot_name = "Drons Team: "
# # Initialize the Telegram bot
# bot = telebot.TeleBot(BOT_TOKEN)

# # Handler for the /start command
# @bot.message_handler(commands=['start'])
# def start(update, context):
#     update.message.reply_text("مرحباً بك ، مالذي تشعر به؟")



# from googletrans import Translator

# # Initialize the Google Translator
# translator = Translator()

# # ... (بقية الكود)

# # Handler for incoming messages
# def handle_message(update, context):
#     user_input = update.message.text

#     # Translate user_input from Arabic to English
#     translated_text = translator.translate(user_input, src='ar', dest='en').text

#     # Check if the user previously responded to more_info_prompt
#     if "waiting_for_response" in context.user_data:
#         handle_more_info_response(update, context, translated_text)  # Handle the response
#     else:
#         # Use the ANN model to predict the disease category and confidence
#         predicted_class, confidence = ann_predict_with_confidence(translated_text)

#         # Translate the response back to Arabic
#         translated_response = translator.translate(f"Predicted Class: {predicted_class} - Probability: {confidence:.4f}", src='en', dest='ar').text

#         update.message.reply_text(translated_response)

#         if confidence >= 0.2:
#             more_info_prompt = "Do you want more details about the predicted disease? (yes/no): "
#             # Translate the more_info_prompt to Arabic
#             translated_more_info_prompt = translator.translate(more_info_prompt, src='en', dest='ar').text
#             update.message.reply_text(translated_more_info_prompt)
#             # Set a state to wait for the user's response
#             context.user_data["waiting_for_response"] = True
#             # Set the predicted_class to the context for later use
#             context.user_data["predicted_class"] = predicted_class
#         else:
#             context.user_data.pop("waiting_for_response", None)  # Clear waiting state if any
#             update.message.reply_text("Can you provide more information about the symptoms you're experiencing?")


# # Handler for processing "yes" or "no" response
# def handle_more_info_response(update, context, translated_text):
#     user_response = update.message.text.strip().lower()

#     if user_response == translator.translate('yes'):
#         # Retrieve the predicted_class from the context
#         predicted_class = context.user_data.get("predicted_class")
#         if predicted_class:
#             more_info = more_iform(predicted_class)
#             # Translate more_info back to Arabic
#             translated_more_info = translator.translate(more_info)
#             update.message.reply_text(translated_more_info)
#         else:
#             update.message.reply_text("أنا آسف ، لم أتمكن من العثور على الفصل المتوقع.")
#     elif user_response == translator.translate('no'):
#         update.message.reply_text(translator.translate("حسنًا ، إذا كان لديك أي أسئلة أخرى ، فلا تتردد في طرحها."))
#     else:
#         update.message.reply_text(translator.translate("الرجاء الرد ' بنعم' أو ' لا' ."))

#     # Clear the waiting state after processing the response
#     context.user_data.pop("waiting_for_response", None)

# # Main function to run the bot
# def main():
#     updater = Updater(token=BOT_TOKEN, use_context=True)
#     dispatcher = updater.dispatcher

#     # Handlers for commands and messages
#     dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
#     dispatcher.add_handler(MessageHandler(Filters.regex(r'^(yes|no)$'), handle_more_info_response))

#     # Start the bot
#     updater.start_polling()
#     updater.idle()

# if __name__ == "__main__":
#     main()






