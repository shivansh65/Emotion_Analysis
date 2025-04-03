# Emotion Detection and Analysis

## Description
This project detects emotions from audio files, keystroke dynamics, and text patterns. It provides detailed analysis reports and motivational messages based on detected emotions.

## Prerequisites
- Python 3.8 or higher
- Install the required dependencies listed below.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/shivansh65/Emotion_Analysis.git
   cd Emotion_Analysis
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies
The following Python libraries are required:
- Flask
- SQLAlchemy
- numpy
- librosa
- torch
- scikit-learn
- matplotlib
- seaborn
- pydub

To install them manually, run:
```bash
pip install flask sqlalchemy numpy librosa torch scikit-learn matplotlib seaborn pydub
```

## Running the Application
1. Ensure the database is initialized:
   ```bash
   python app.py
   ```
   This will create the necessary SQLite database.

2. Start the Flask application:
   ```bash
   python app.py
   ```
   The application will run on `http://127.0.0.1:5001`.

## Testing
- Test the `/check_ffmpeg` endpoint to ensure FFmpeg is installed:
  ```bash
  curl http://127.0.0.1:5001/check_ffmpeg
  ```
- Upload audio files or interact with the keystroke analysis feature via the web interface.

## Additional Notes
- Ensure FFmpeg is installed on your system. You can install it via:
  ```bash
  sudo apt install ffmpeg  # On Ubuntu
  brew install ffmpeg      # On macOS
  ```
- Place the `emotion_voice_model.pth` and `label_encoder_classes.npy` files in the project directory.

## Repository
This project is hosted on GitHub: [Emotion Analysis](https://github.com/shivansh65/Emotion_Analysis)
