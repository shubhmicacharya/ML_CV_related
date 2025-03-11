import nltk
import random
import pyttsx3

engine = pyttsx3.init()

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Sample responses
responses = {
    "greeting": ["Hello! How can I assist you today?", "Hi there! What can I do for you?", "Greetings!"],
    "goodbye": ["Goodbye! Have a great day!", "Bye! Take care.", "See you later!"],
    "thanks": ["You're welcome!", "Happy to help!", "No problem at all!"],
    "default": ["I'm sorry, I didn't understand that.", "Can you rephrase that?",
                "I'm here to help, but I didn't catch that."],
    "how_are_you": ["I'm just a bot, but I'm doing great! How about you?", "I'm good, thanks for asking!",
                    "I'm just a program, but I'm functioning well!"],
    "help": ["How can I assist you today?", "What do you need help with?", "I'm here to help you, just ask!"]
}

# Keywords for classification
keywords = {
    "greeting": ["hello", "hi", "hey", "greetings"],
    "goodbye": ["bye", "goodbye", "see you", "later"],
    "thanks": ["thank", "thanks", "thank you"],
    "how_are_you": ["how are you", "how's it going", "how do you do"],
    "help": ["help", "assist", "support"]
}


# Function to classify the user's input
def classify_input(user_input):
    user_tokens = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(user_input)]
    for intent, words in keywords.items():
        if any(word in user_tokens for word in words):
            return intent
    return "default"


# Chatbot logic
def chatbot():
    engine.say("Hi! I'm a simple chatbot. Type 'exit' to end the conversation.")
    engine.runAndWait()
    print("Chatbot: Hi! I'm a simple chatbot. Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            engine.say("Goodbye!")
            engine.runAndWait()
            break

        intent = classify_input(user_input)
        response = random.choice(responses[intent])
        print(f"Chatbot: {response}")
        engine.say(response)
        engine.runAndWait()


# Run the chatbot
if __name__ == "__main__":
    chatbot()
