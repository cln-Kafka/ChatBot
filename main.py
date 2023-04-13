# Importing required libraries and modules.

import nltk
# NLTK (Natural Language Toolkit) is a Python library that offers functionalities for tasks such as
# text classification, tokenization, stemming, part-of-speech tagging, and sentiment analysis, among others.

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# A stemmer is a tool used in NLP to reduce words to their base or
# root form (known as a stem) by removing any suffixes or prefixes.
# it's used to improve the accuracy and efficiency of the chatbot's natural language processing.

import numpy as np
# NumPy arrays are used for storing and manipulating large amounts of numerical data,
# and they can be easily converted to and from TensorFlow tensors using built-in conversion functions.

import tensorflow as tf
# TensorFlow: open-source (ML) library developed by Google.
# that helps a lot in building and training various types of (ML) models, including neural networks.
# It is widely used for building chatbots.

import tflearn
# TFlearn is a high-level deep learning library built on top of TensorFlow.
# In this code, TFlearn can be used to build and train neural network models for
# tasks such as intent recognition/entity recognition, or sentiment analysis.

import json
# JSON (JavaScript Object Notation) data.
# In our context, the "json" module can be used to encode and decode JSON data from external sources, such as APIs or web services...
# We are using it for to deal with the intents file. 

import random
# The "random" module in Python provides functions for working with random numbers and selections or shuffle sequences and more.
# In this context, the "random" module can be used to introduce variability or randomness into the chatbot's responses or behavior.
# In other words, if there are multiple responses defined to a list of patterns, all lying under the same tag. It will choose random response
# from them. ex: saying "hello" or "hi" or may be "hi there, how can i help you?"

import pickle
# Pickle can be used to save trained models or other important data structures to disk,
# so that they can be easily loaded and reused later.
# This can be useful for maintaining state across multiple sessions or for offline processing.

# colorama to print colored text
from colorama import Fore

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

# in line 52 and 53, we are loading the intents file that will be used to train our model.
with open("intents.json") as file:
    data = json.load(file)

try:
    # at the first time to train the model, we save the data like words, labels, training, and output variables
    # in a pickle file (this is represented below AFTER the data is ready to be stored).
    # but after the first time, we don't need to retrain the model every time we run it so we are
    # checking if the pickle file exists and contain the data.
    # if yes, it loads the contents of the file into the variables words, labels, training, and output.
    # if no, retrain the model.
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = [] # a list of all of the words of our patterns
    labels = [] # a list of all of our tags
    docs_x = [] # a list of all different patterns
    docs_y = [] # a list that stores the corresponding tag for each pattern in docs_x

    for intent in data["intents"]: # looping on each intent of the intents file
        for pattern in intent["patterns"]: # looping on each pattern of the current intent specified from the higher loop.

            # word_tokenize -> function from "nltk" to break up a piece of string into smaller units (tokens).
            # e.g. "what is your name?" -->> ["what", "is", "your", "name", "?"]
            # Storing these words in a list called: words_of_patterns
            words_of_patterns = nltk.word_tokenize(pattern)

            # Extending words by words_of_patterns
            words.extend(words_of_patterns)

            # Adding/Appending each unit or token as a pattern in a list called ((docs_x))
            docs_x.append(words_of_patterns)

            # Classifying our patterns/identifying what tag that the pattern belongs to
            # Storing it in a list called ((docs_y))
            docs_y.append(intent["tag"])

        # storing the tags in a list called labels
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Stemming all the words in the "words" list and removing all duplicate lists.
    # Using (w.lower) to convert all words to lowercase form.
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]

    # set() -> takes all the words and makes sure there is no duplicate (that is a property of sets)
    # list(set()) -> convert the set into list.
    # sorted() -> Return a new list containing all items from the iterable in ascending order.
    words = sorted(list(set(words)))

    labels = sorted(labels)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    # Converting the inputs and outputs from strings to numbers so that the model can deal with them.
    # in the following code, the "one-hot encoding" technique is used.
    # one-hot encodings is a process of representing categorical data as binary vectors with a single "hot" or "on" bit and all others "off".
    training = []
    output = []

    # out_empty is a list of zeros with a length equal to the number of tags -> [0, 0, 0]
    out_empty = [0 for _ in range(len(labels))]

    # "bag" is a list which will contain the numerical representation of the sentence.
    # The length of this list is the same as the number of words in the words list.
    for x, doc in enumerate(docs_x):
        bag = []

        words_of_patterns = [stemmer.stem(w) for w in doc]

    # For each word in the sentence (input), the function checks if it is in the words list.
    # If it is, the corresponding element in the bag list is set to 1. If it's not, that element remains 0.
        for w in words:
            if w in words_of_patterns:
                bag.append(1)
            else:
                bag.append(0)

        # output_row is created by copying out_empty using the [:] syntax, which creates a new list with the same values as out_empty.
        # This is done to avoid creating a reference to the original out_empty list.
        output_row = out_empty[:]
        # The value in output_row corresponding to the index of the current tag is then set to 1.
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # Finally, the bag list is returned as a NumPy array, which can be used as input to a machine learning model.
    training = np.array(training)
    output = np.array(output)

#Saving the data like words, labels, training, and output variables in a pickle file
with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

# For our purposes we will use a fairly standard feed-forward neural network with two hidden layers.
# The goal of our network will be to look at a bag of words and give a class that they belong too (one of our tags from the JSON file).
tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)

# in simple words, softmax is a mathematical function that converts a vector of real numbers into a probability distribution.
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


# The following segment attempts to load a pre-trained model from the file "model.tflearn".
# If it fails to load the model, it creates a new neural network with three fully connected layers
# and a softmax activation function for the output layer, using the training and output data that were created earlier.
# It then trains this new model on the training data for 3000 epochs with a batch size of 8 and shows the training metrics.
# Finally, it saves the trained model to a file named "model.tflearn".
# This ensures that the model is trained and available for use, even if a pre-trained model does not exist or fails to load.

try:
    model.load("model.tflearn")
except:
    tf.compat.v1.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch = 3000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


# Converting the input sentence into a bag of words after applying tokenizing and stemming.
# And then comparing this input with the words list that contains the words of all of out patterns in the json file.
def bag_of_words(sentence, words):
    bag = [0 for _ in range(len(words))]

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    for word in sentence_words:
        for i, w in enumerate(words):
            if w == word:
                bag[i] = 1

    return np.array(bag)

# ---------------------------------------------------------------------------

# This function is the main function that runs the chatbot.
# It first prompts the user to start a conversation by typing a message.
# It then takes the user input, applies the bag_of_words function to convert the input into a bag of words,
# and uses the model.predict function to predict the tag of the user input based on the trained model.

# If the predicted tag has a confidence score of more than 0.7,
# the function looks for a response associated with the tag in the data dictionary
# and randomly selects and prints one of the responses.

# If the confidence score is less than 0.7,
# the function prints a message indicating that it did not understand the user input and prompts the user to try again.

# The conversation continues until the user types "quit".def chat():
 
def chat():
    print(Fore.GREEN + "Start talking with the bot! (type quit to stop)")
    while True:
        inp = input(Fore.WHITE + "You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
        
            print(Fore.GREEN + random.choice(responses))
        else:
            print(Fore.GREEN + "I didn't get that. Please, try again.")

# calling the chat function
chat()