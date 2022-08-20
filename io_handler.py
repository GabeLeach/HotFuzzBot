import random 
import json
import pickle
import numpy as np
import nltk
from  nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model


lemmatizer = WordNetLemmatizer()

with open("intents.json", "r") as read_file:
  #  print(read_file.read())
    intents = json.loads(read_file.read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.2
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        #return the list in order of most probable
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list
    
def get_response(intents_list, intents_json):
    #tag = most probable intent (position 0 being the most probable)
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    #searching through the list of intents
    for i in list_of_intents:

        if i['tag'] == tag:
            #print a random string from the responses list
            result = random.choice(i['responses'])
            break
    return result
      

def user_name():
    #get users name input
    name = input("")
    name_stored = name    
    #ask to verify, important in chatbot and voicebots as they could accept incorrect information.
    print("Please confirm your name is " + name_stored + " by staying 'yes' or 'no'.")
    verify = input("")
    #if verified tell the user and return the number to be stored
    if verify in ["Yes", "yes"]:
        name_true = name_stored
        print("Name confirmed.\n")
        return name_true
    else:
        print("Please enter your name.\n")
        user_name()

bot_name = "Hot Fuzz Bot"      

#GUI function
def get_from_gui(msg):
    user_input = msg  
    ints = predict_class(user_input)
    res = get_response(ints, intents)
    return(res)