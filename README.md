# ChatBot
This is my first task as a machine learning intern at SYNC INTERN's
The provided code is a Python script that builds a simple chatbot using TensorFlow and TFLearn libraries.
The chatbot is trained on a dataset of pre-defined intents and responds to user inputs based on the identified intent.

The code uses NLTK for tokenization and stemming, and the dataset is stored in a JSON file called "intents.json".
The script loads the data, tokenizes the patterns and creates a "bag of words" for each pattern.
A "bag of words" is a representation of text that describes the occurrence of words within a document. Each bag of words is associated with an intent or a tag.

The script then trains a neural network using TFLearn with the bag of words as input and the associated tag as output.
The trained model is saved to a file named "model.tflearn".

Finally, the script defines a function called "chat()" that allows users to interact with the chatbot.
The function prompts the user to input a message and predicts the intent of the message using the trained model.
If the predicted intent has a high confidence level, the chatbot responds with a randomly selected response from the dataset.
If the confidence level is low, the chatbot prompts the user to try again.

Overall, this script provides a simple framework for building and training a chatbot that can understand user intents and respond accordingly.
