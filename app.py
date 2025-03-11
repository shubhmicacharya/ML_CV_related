from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3
import speech_recognition as sr
import threading
import webbrowser

# Initialize Flask app
app = Flask(__name__)

# Initialize Text-to-Speech (TTS) engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech rate
engine.setProperty('volume', 1.0)  # Set volume


def speak(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()


# Initialize Speech Recognition
recognizer = sr.Recognizer()
wake_word = "jarvis"  # Wake word for the assistant


def listen_for_wake_word():
    """Continuously listens for the wake word."""
    print("Listening for the wake word 'Jarvis'...")
    while True:
        with sr.Microphone() as source:
            try:
                recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
                audio = recognizer.listen(source)  # Capture audio

                # Recognize the speech
                command = recognizer.recognize_google(audio).lower()
                print(f"Heard: {command}")

                # Check for wake word
                if wake_word in command:
                    print("Wake word detected! Opening assistant page.")
                    speak("Hello! I am Jarvis. Opening the assistant page.")
                    # Trigger the page through Flask
                    webbrowser.open_new("http://127.0.0.1:5000/assistant")
            except sr.UnknownValueError:
                print("Didn't catch that. Waiting for the wake word...")
            except sr.RequestError as e:
                print(f"Speech recognition service error: {e}")


# Load dataset
df = pd.read_csv('data_moods.csv')  # Ensure this matches your CSV file format

# Load pickled TF-IDF vectorizer and matrix
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)


@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')


@app.route('/assistant')
def assistant():
    """Render the AI Assistant page."""
    return render_template('assistant.html')


@app.route('/suggest', methods=['GET'])
def suggest():
    """Provide song suggestions based on a query."""
    query = request.args.get('query', '').strip().lower()

    if not query:
        return jsonify({'error': 'Empty query'}), 400

    # Ensure the query is transformed to the same format used for training
    query_vec = tfidf_vectorizer.transform([query])
    query_sim = cosine_similarity(query_vec, tfidf_matrix)

    # Get top 5 most similar songs
    top_n = 5
    similar_indices = query_sim.argsort()[0][-top_n:][::-1]

    recommendations = []
    for idx in similar_indices:
        song = df.iloc[idx]
        song_name = song['name']
        artist_name = song['artist']
        youtube_link = f"https://www.youtube.com/results?search_query={song_name} {artist_name}"

        recommendations.append({
            'name': song_name,
            'artist': artist_name,
            'youtube_link': youtube_link
        })

    return jsonify(recommendations)



@app.route('/chat', methods=['POST'])
def chat():
    """Chat interface for Jarvis."""
    try:
        user_message = request.json.get('message', '').strip().lower()
        print(f"User Message: {user_message}")

        if "hello" in user_message:
            return jsonify({"response": "Hello! How can I assist you today?"})

        # Define mood-based keywords
        keywords = {
            "happy": ["happy", "joyful", "excited", "cheerful", "elated", "glad"],
            "sad": ["sad", "down", "depressed", "melancholy", "blue", "gloomy"],
            "energetic": ["energetic", "active", "motivated", "pumped", "excited", "hyper"],
            "calm": ["calm", "relaxed", "peaceful", "chill", "serene", "laid back"],
        }

        detected_mood = next((mood for mood, words in keywords.items() if any(word in user_message for word in words)), None)

        if detected_mood:
            recommendations = df[df["mood"].str.strip().str.lower() == detected_mood]
            if recommendations.empty:
                return jsonify({"response": f"Sorry, I couldn't find any songs for {detected_mood} mood."})

            top_5_songs = recommendations.head(5)
            songs = []
            for _, row in top_5_songs.iterrows():
                name = row['name']
                artist = row['artist']
                youtube_link = f"https://www.youtube.com/results?search_query={name} {artist}"
                songs.append({"name": name, 
                              "artist": artist, 
                              "youtube_link": youtube_link})

            return jsonify({
                "response": f"Here are the top 5 {detected_mood} mood songs:",
                "songs": songs
            })

        return jsonify({"response": "I'm not sure how to respond to that."})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred."}), 500



if __name__ == '__main__':
    assistant_thread = threading.Thread(target=listen_for_wake_word)
    assistant_thread.daemon = True
    assistant_thread.start()

    app.run(debug=True)
