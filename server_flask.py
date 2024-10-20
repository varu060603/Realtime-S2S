
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_socketio import SocketIO, emit
import random
import base64
from datetime import datetime
import threading
import os
import torch
from gtts import gTTS
import noisereduce as nr
import numpy as np
import soundfile as sf
from googletrans import Translator
import pandas as pd
import atexit
import speech_recognition as sr
from faster_whisper import WhisperModel

# Set the environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model_size = "large-v3"
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if torch.cuda.is_available() else "int8"
faster_model = WhisperModel(model_size, device=device, compute_type=compute_type)
translator = Translator()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

sample_output_audio_dir = 'C:/WORK/varshitwork/intel_new/outputs'
noise_reduced_audio_dir = "C:/WORK/varshitwork/intel_new/noise_reduced"
input_audio_dir = "C:/WORK/varshitwork/intel_new/inputs"

df = pd.DataFrame(columns=['chunk_id', 'received_time', 'process_start_time', 'process_end_time', 'transcription'])

def save_dataframe():
    df.to_csv('/home/sura/bhashini-seamless-realtime/my_model/timestamps.csv', index=False)

atexit.register(save_dataframe)

lock = threading.Lock()

def saveB64audio(b64_chunk):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{input_audio_dir}/{timestamp}_{random.randint(0, 100)}.wav"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as wav_file:
        decode_string = base64.b64decode(b64_chunk)
        wav_file.write(decode_string)
    return filename

def reduce_noise(input_audio_path):
    data, rate = sf.read(input_audio_path)
    reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=1)
    os.makedirs(noise_reduced_audio_dir, exist_ok=True)
    noise_reduced_path = os.path.join(noise_reduced_audio_dir, os.path.basename(input_audio_path))
    sf.write(noise_reduced_path, reduced_noise, rate)
    return noise_reduced_path

def run_s2tt_google(input_audio):
    recognizer = sr.Recognizer()
    with sr.AudioFile(input_audio) as source:
        audio = recognizer.record(source)
    try:
        transcription = recognizer.recognize_google(audio)
        detected_language = "en"
    except sr.UnknownValueError:
        transcription = None
        detected_language = "nn"
    return transcription, detected_language

def run_s2tt_whisper(input_audio):
    segments, detected_language = faster_model.transcribe(input_audio, vad_filter=True)
    transcription = " ".join([segment.text for segment in segments])
    return transcription, detected_language.language

def translate_text(text, language='en'):
    start_time = datetime.now()
    translation = translator.translate(text, dest=language)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    return translation.text, start_time, end_time, duration

def run_tts(text, language='en'):
    translated_text, translate_start_time, translate_end_time, translate_duration = translate_text(text, language)
    tts_start_time = datetime.now()
    tts = gTTS(text=translated_text, lang=language)
    output_path = f"{sample_output_audio_dir}/chunk_{random.randint(0, 100)}.mp3"
    tts.save(output_path)
    tts_end_time = datetime.now()
    tts_duration = (tts_end_time - tts_start_time).total_seconds()
    return output_path, translated_text, translate_start_time, translate_end_time, translate_duration, tts_start_time, tts_end_time, tts_duration

@app.route("/")
def index():
    return "Audio Processing Server"

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_from_directory(input_audio_dir, filename)

@socketio.on("connect")
def test_connect():
    print("Client connected")

@socketio.on("disconnect")
def test_disconnect():
    print("Client disconnected")

@socketio.on("audio_stream")
def handle_audio(data):
    print("\nReceived a chunk\n")

    with lock:
        chunk_id = random.randint(0, 100)
        received_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        audio_chunk = data["audio"]
        saved_filename = saveB64audio(audio_chunk)

        process_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        noise_reduction_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        noise_reduced_filename = reduce_noise(saved_filename)

        noise_reduction_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        noise_reduction_duration = (datetime.strptime(noise_reduction_end_time, "%Y-%m-%d %H:%M:%S.%f") - datetime.strptime(noise_reduction_start_time, "%Y-%m-%d %H:%M:%S.%f")).total_seconds()

        transcription_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        model = data.get("model", "google")

        if model == "faster_whisper":
            transcription, detected_language = run_s2tt_whisper(noise_reduced_filename)
        else:
            transcription, detected_language = run_s2tt_google(noise_reduced_filename)

        transcription_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        transcription_duration = (datetime.strptime(transcription_end_time, "%Y-%m-%d %H:%M:%S.%f") - datetime.strptime(transcription_start_time, "%Y-%m-%d %H:%M:%S.%f")).total_seconds()

        if transcription and detected_language != 'nn':
            tts_path, translated_text, translate_start_time, translate_end_time, translate_duration, tts_start_time, tts_end_time, tts_duration = run_tts(transcription, language=data.get("language", "en"))

            process_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            process_duration = (datetime.strptime(process_end_time, "%Y-%m-%d %H:%M:%S.%f") - datetime.strptime(process_start_time, "%Y-%m-%d %H:%M:%S.%f")).total_seconds()

            with open(tts_path, "rb") as processed_file:
                processed_data = processed_file.read()

            print("Audio processed, sending event")
            emit(
                "processed_audio",
                {
                    "audio": base64.b64encode(processed_data).decode(),
                    "transcription": transcription,
                    "chunk_id": chunk_id,
                    "received_time": received_time,
                    "process_start_time": process_start_time,
                    "noise_reduction_start_time": noise_reduction_start_time,
                    "noise_reduction_end_time": noise_reduction_end_time,
                    "noise_reduction_duration": noise_reduction_duration,
                    "transcription_start_time": transcription_start_time,
                    "transcription_end_time": transcription_end_time,
                    "transcription_duration": transcription_duration,
                    "translate_start_time": translate_start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "translate_end_time": translate_end_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "translate_duration": translate_duration,
                    "tts_start_time": tts_start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "tts_end_time": tts_end_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "tts_duration": tts_duration,
                    "process_duration": process_duration,
                    "input_filename": os.path.basename(saved_filename),
                },
                binary=False,
            )
        else:
            print("No transcription available")
            emit(
                "processed_audio",
                {
                    "audio": None,
                    "transcription": "No transcription available"
                },
                binary=False,
            )

@socketio.on("translate_text")
def handle_translation(data):
    text_to_translate = data.get("text", "")
    selected_language = data.get("language", "en")
    
    translated_text = translator.translate(text_to_translate, dest=selected_language).text

    emit("translated_text", {"translated_text": translated_text})

@socketio.on("text_to_speech")
def handle_text_to_speech(data):
    text_to_speak = data.get("text", "")
    selected_language = data.get("language", "en")
    
    tts_path, translated_text, translate_start_time, translate_end_time, translate_duration, tts_start_time, tts_end_time, tts_duration = run_tts(text_to_speak, language=selected_language)
    
    with open(tts_path, "rb") as audio_file:
        audio_data = audio_file.read()
    
    emit("synthesized_speech", {"audio": base64.b64encode(audio_data).decode()})

if __name__ == "__main__":
    socketio.run(app, debug=False, port=7860, host="0.0.0.0", log_output=True)
