""" Importing Libraries 🙂 """
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import librosa
import pyaudio
import wave

"""Step 3: Load pre-trained LLM model and tokenizer"""

model_name = "facebook/wav2vec2-base-960h"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)

"""Step 4: Define a function to record audio input"""

def record_audio(duration=5, sample_rate=22050):
    print("Recording audio...")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)
    frames = []
    for i in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    return b''.join(frames)

"""Step 5: Define a function to preprocess audio input"""

def preprocess_audio(audio_data):
    audio_data = np.frombuffer(audio_data, dtype=np.int16)
    audio_data = librosa.resample(audio_data, orig_sr=22050, target_sr=16000)
    audio_data = librosa.util.normalize(audio_data)
    return audio_data

"""Step 6: Define a function to generate response ✌️"""

def generate_response(audio_data):
    inputs = tokenizer.encode_plus(
        audio_data,
        return_tensors="pt",
        max_length=1024,
        padding="max_length",
        truncation=True,
    )
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    logits = outputs.logits
    response = torch.argmax(logits)
    return response

"""Step 7: Create a main function to integrate everything"""

def main():
    while True:
        audio_data = record_audio()
        audio_data = preprocess_audio(audio_data)
        response = generate_response(audio_data)
        print(f"Response: {response}")

"""Step 8: Run the main function"""
main()
