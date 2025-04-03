from flask import Flask, render_template, request, flash, jsonify, redirect, url_for
import numpy as np
import librosa
import torch
from sklearn.preprocessing import LabelEncoder
from model import EmotionDetectionModel
import json
import random
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
from flask_sqlalchemy import SQLAlchemy
import re
from datetime import datetime
import subprocess

# Ensure you have pydub installed
# You can install it using pip:
# pip install pydub

from pydub import AudioSegment
matplotlib.use('Agg')  # Use the Agg backend for non-GUI environments

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Use a relative path for the database
db = SQLAlchemy(app)

# Load model and encoder
model = EmotionDetectionModel(num_classes=4)
model.load_state_dict(torch.load("emotion_voice_model.pth"))
model.eval()

le = LabelEncoder()
le.classes_ = np.load("label_encoder_classes.npy", allow_pickle=True)

def convert_to_wav(file):
    """Convert any audio file to .wav format."""
    try:
        audio = AudioSegment.from_file(file)
        temp_path = "static/temp_recording.wav"
        audio.export(temp_path, format="wav")
        return temp_path
    except Exception as e:
        print(f"Error converting file to .wav: {e}")
        raise

def extract_mfcc_features(audio, sr):
    """Extract MFCC features from audio."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
    if mfcc.shape[1] != 20:
        mfcc = torch.nn.functional.interpolate(mfcc.unsqueeze(0), size=(20,), mode='linear', align_corners=False).squeeze(0)
    return mfcc

def predict_emotion(mfcc):
    """Predict emotion from MFCC features."""
    with torch.no_grad():
        output = model(mfcc.unsqueeze(0))
        _, predicted = torch.max(output, 1)
        emotion = le.inverse_transform(predicted.numpy())[0]
    return emotion

def generate_analysis_report(audio, sr):
    """Generate a detailed analysis report for the audio."""
    duration = librosa.get_duration(y=audio, sr=sr)
    interval_length = 4  # Length of each interval in seconds
    intervals = np.arange(0, duration, interval_length).tolist()
    if intervals[-1] != duration:
        intervals.append(duration)
    
    emotions = []
    current_emotion = None
    emotion_intervals = []

    for start, end in zip(intervals[:-1], intervals[1:]):
        segment = audio[int(start * sr):int(end * sr)]
        mfcc = extract_mfcc_features(segment, sr)
        emotion = predict_emotion(mfcc)
        
        if emotion != current_emotion:
            if current_emotion is not None:
                emotion_intervals.append((current_emotion, start, end))
            current_emotion = emotion
    
    if current_emotion is not None:
        emotion_intervals.append((current_emotion, intervals[-2], duration))

    # Combine intervals with the same emotion
    combined_intervals = []
    for emotion, start, end in emotion_intervals:
        if combined_intervals and combined_intervals[-1][0] == emotion:
            combined_intervals[-1] = (emotion, combined_intervals[-1][1], end)
        else:
            combined_intervals.append((emotion, start, end))

    # Generate emotion intensity graph using seaborn
    plt.figure(figsize=(10, 4))
    sns.set(style="whitegrid")
    for emotion, start, end in combined_intervals:
        plt.plot([start, end], [emotion, emotion], marker='o', label=emotion)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Emotion')
    plt.title('Emotion Changes Over Time')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # Save the report
    report_path = 'static/detailed_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write('Detailed Analysis Report\n')
        f.write('========================\n')
        for emotion, start, end in combined_intervals:
            f.write(f'Time Interval {round(start, 2)}-{round(end, 2)} seconds: {emotion}\n')

    return {
        'time_intervals': [{'start': round(start, 2), 'end': round(end, 2)} for _, start, end in combined_intervals],
        'emotion_intensity': [emotion for emotion, _, _ in combined_intervals],
        'report_url': url_for('static', filename='detailed_analysis_report.txt'),
        'graph_url': f'data:image/png;base64,{image_base64}'
    }

def analyze_text(text):
    """Analyze the input text for emotional indicators."""
    positive_words = ["happy", "joy", "love", "excited", "great"]
    negative_words = ["sad", "angry", "hate", "upset", "bad"]
    positive_count = sum(1 for word in text.split() if word.lower() in positive_words)
    negative_count = sum(1 for word in text.split() if word.lower() in negative_words)
    return positive_count, negative_count

def calculate_keystroke_features(keystrokes):
    """Calculate keystroke features from the collected data."""
    dwell_times = [k['dwell_time'] for k in keystrokes]
    flight_times = [k['flight_time'] for k in keystrokes if k['flight_time'] is not None]
    down_to_down_times = [k['down_to_down_time'] for k in keystrokes if k['down_to_down_time'] is not None]

    features = {
        'typing_speed': len(keystrokes) / (keystrokes[-1]['timestamp'] - keystrokes[0]['timestamp']),
        'min_dwell': min(dwell_times),
        'min_flight': min(flight_times) if flight_times else 0,
        'min_d2d': min(down_to_down_times) if down_to_down_times else 0,
        'mode_dwell': max(set(dwell_times), key=dwell_times.count),
        'mode_flight': max(set(flight_times), key=flight_times.count) if flight_times else 0,
        'mode_d2d': max(set(down_to_down_times), key=down_to_down_times.count) if down_to_down_times else 0,
        'std_dwell': np.std(dwell_times),
        'std_flight': np.std(flight_times) if flight_times else 0,
        'std_d2d': np.std(down_to_down_times) if down_to_down_times else 0,
        'backspace_rate': sum(1 for k in keystrokes if k['key'] == 'Backspace') / (keystrokes[-1]['timestamp'] - keystrokes[0]['timestamp']) * 60000
    }
    return features

def detect_emotion_from_keystrokes(keystroke_features, text_analysis):
    """Detect emotion based on keystroke features and text analysis."""
    emotion = "neutral"

    # Keystroke dynamics-based conditions
    if keystroke_features['backspace_rate'] > 5:
        emotion = "anxious"
    elif keystroke_features['typing_speed'] < 20 and keystroke_features['mode_dwell'] > 200:
        emotion = "sad"
    elif keystroke_features['typing_speed'] > 60 and keystroke_features['min_flight'] < 100:
        emotion = "angry"
    elif keystroke_features['typing_speed'] < 10 or keystroke_features['std_d2d'] > 150:
        emotion = "tired"
    elif keystroke_features['std_flight'] > 150:
        emotion = "fearful"

    # Text pattern-based conditions
    if text_analysis['grammatical_errors'] > 3:
        emotion = "confused"
    elif text_analysis['repetitive_words'] > 3:
        emotion = "anxious"
    elif text_analysis['negative_words'] > 0 and text_analysis['negations'] > 0:
        emotion = "sad"
    elif text_analysis['aggressive_words'] > 2:
        emotion = "angry"
    elif text_analysis['short_sentences'] > 0:
        emotion = "disgust"

    return emotion

def analyze_text_patterns(text):
    """Analyze text patterns for emotional indicators."""
    grammatical_errors = len(re.findall(r'\b(?:is|are|was|were|has|have|had|do|does|did|will|would|shall|should|can|could|may|might|must|ought)\b', text))
    repetitive_words = len([word for word in text.split() if text.split().count(word) > 3])
    negative_words = len([word for word in text.split() if word.lower() in ["sad", "hopeless", "angry", "hate", "bad"]])
    negations = len([word for word in text.split() if word.lower() in ["not", "never", "no"]])
    aggressive_words = len([word for word in text.split() if word.lower() in ["hate", "stupid", "idiot", "fool", "dumb"]])
    short_sentences = len([sentence for sentence in text.split('.') if len(sentence.split()) < 5])

    return {
        'grammatical_errors': grammatical_errors,
        'repetitive_words': repetitive_words,
        'negative_words': negative_words,
        'negations': negations,
        'aggressive_words': aggressive_words,
        'short_sentences': short_sentences
    }

@app.route('/check_ffmpeg')
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return jsonify({'status': 'success'})
    except (subprocess.CalledProcessError, FileNotFoundError):
        return jsonify({'status': 'error'}), 500

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        pass
    
    detected_emotion = request.args.get("emotion")
    return render_template("home.html", detected_emotion=detected_emotion)

@app.route("/audio", methods=["GET", "POST"])
def audio():
    emotion = None  # Reset emotion to None at the start of the request
    audio_file = None  # Add this line
    motivational_message = None  # Add this line
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            try:
                print("File uploaded successfully:", file.filename)

                # Convert to .wav format if necessary
                temp_path = convert_to_wav(file)
                audio_file = temp_path  # Add this line

                # Load and process the audio file
                print("Loading audio file...")
                
                audio, sr = librosa.load(temp_path, sr=22050)
                print("Audio file loaded. Extracting MFCC features...")

                # Extract MFCC features
                mfcc = extract_mfcc_features(audio, sr)
                print("MFCC tensor shape:", mfcc.shape)

                # Predict emotion
                emotion = predict_emotion(mfcc)
                print("Predicted emotion:", emotion)
                
                # Add motivational message
                motivational_messages = {
                    "angry": "Take a deep breath. Everything will be okay.",
                    "stressed": "Remember to take breaks and relax. You've got this!",
                    "anxious": "Stay positive and keep moving forward. You're doing great!",
                    "sad": "It's okay to feel sad sometimes. Things will get better."
                }
                if emotion in motivational_messages:
                    motivational_message = motivational_messages[emotion]

                flash(f"Emotion detected: {emotion}", "success")
                return render_template("audio.html", emotion=emotion, audio_file=audio_file, motivational_message=motivational_message)  # Pass audio_file and motivational_message
            except Exception as e:
                flash(f"An error occurred while processing the audio file: {e}", "danger")
                return render_template("audio.html", emotion=None)
        else:
            flash("Invalid file format. Please upload a media file.", "danger")
            return render_template("audio.html", emotion=None)
    return render_template("audio.html", emotion=emotion, audio_file=audio_file, motivational_message=motivational_message)  # Pass audio_file and motivational_message

@app.route("/keystroke", methods=["GET", "POST"])
def keystroke():
    random_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The early bird catches the worm."
    ]
    selected_text = random.choice(random_texts)
    if request.method == "POST":
        data = request.get_json()
        keystrokes = data.get("keystrokes")
        text = data.get("text")

        # Calculate keystroke features
        keystroke_features = calculate_keystroke_features(keystrokes)

        # Analyze the text for emotional indicators
        text_analysis = analyze_text_patterns(text)

        # Detect emotion based on keystroke features and text analysis
        emotion = detect_emotion_from_keystrokes(keystroke_features, text_analysis)

        # Detailed emotion report
        emotion_report = {
            "emotion": emotion,
            "keystroke_features": keystroke_features,
            "text_analysis": text_analysis
        }

        return jsonify(emotion_report)
    return render_template("keystroke.html", selected_text=selected_text)

@app.route("/start_keystroke_analysis", methods=["POST"])
def start_keystroke_analysis():
    duration = request.form.get("duration", type=int)
    return render_template("keystroke_analysis.html", duration=duration)

@app.route("/detailed_analysis", methods=["POST"])
def detailed_analysis():
    data = request.get_json()
    file_path = data.get("file")
    if file_path:
        try:
            audio, sr = librosa.load(file_path, sr=22050)
            analysis_report = generate_analysis_report(audio, sr)
            return jsonify(analysis_report)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "File not found"}), 400

@app.route("/detect_recorded_emotion", methods=["POST"])
def detect_recorded_emotion():
    file = request.files.get("file")
    if file:
        try:
            temp_path = convert_to_wav(file)
            audio, sr = librosa.load(temp_path, sr=22050)
            mfcc = extract_mfcc_features(audio, sr)
            emotion = predict_emotion(mfcc)
            
            # Add motivational message
            motivational_messages = {
                "angry": "Take a deep breath. Everything will be okay.",
                "stressed": "Remember to take breaks and relax. You've got this!",
                "anxious": "Stay positive and keep moving forward. You're doing great!",
                "sad": "It's okay to feel sad sometimes. Things will get better."
            }
            motivational_message = motivational_messages.get(emotion, "")

            return jsonify({"emotion": emotion, "motivational_message": motivational_message})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "File not found"}), 400

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Ensure the database and tables are created
    app.run(debug=True, port=5001)