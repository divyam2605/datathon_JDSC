import subprocess
import librosa
import pandas as pd
import pyworld as pw  
from decimal import Decimal, getcontext
import aubio
import numpy as np
from scipy.signal import find_peaks
from pydub import AudioSegment
from scipy.stats import linregress

def convert_to_wav(input_file_path, output_file_path):
    ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
    subprocess.run([ffmpeg_path, "-i", input_file_path, "-ac", "2", "-ar", "44100", output_file_path])
    print(f"Conversion successful. Saved as {output_file_path}")

def calculate_mdvp_jitter_abs(file_path, samplerate, win_size):
    hop_size = win_size // 2

    source = aubio.source(file_path, samplerate, hop_size)
    pitch_o = aubio.pitch("yin", win_size, hop_size, samplerate)

    pitch_periods = []

    while True:
        samples, read = source()
        pitch = pitch_o(samples)[0]
        pitch_periods.append(pitch)
        if read < hop_size:
            break

    absolute_diff = np.abs(np.diff(pitch_periods))
    mdvp_jitter_abs = np.mean(absolute_diff/1000000)

    return mdvp_jitter_abs


def calculate_mdvp_jitter(file_path):
    getcontext().prec = 10  
    samplerate =44100  
    win_s = 4 # Window size
    hop_s = win_s // 2  # Hop size
    source = aubio.source(file_path, samplerate, hop_s)
    pitch_o = aubio.pitch("yin", win_s, hop_s, samplerate)
    pitch_periods = []
    total_frames = 0
    while True:
        samples, read = source()
        pitch = pitch_o(samples)[0]
        pitch_periods.append(Decimal(str(pitch)))
        total_frames += read
        if read < hop_s:
            break
    absolute_diff = [abs(p2 - p1) for p1, p2 in zip(pitch_periods, pitch_periods[1:])]

    avg_absolute_diff = sum(absolute_diff) / Decimal(len(absolute_diff))
    avg_pitch_period = sum(pitch_periods) / Decimal(len(pitch_periods))
    mdvp_jitter_percentage = (avg_absolute_diff / (avg_pitch_period*100)) 

    return mdvp_jitter_percentage

def mdvp_ppq(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    ppq_index = np.mean(np.abs(np.diff(np.diff(rms))))
    return ppq_index

def calculate_shimmer_features(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)
    rms = librosa.feature.rms(y=y)[0]

    # Estimate pitch using harmonic component
    harmonic = librosa.effects.harmonic(y)
    pitches, voiced_flag, voiced_probs = librosa.pyin(y=harmonic, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    max_magnitude_index = np.argmax(voiced_probs)
    pitch = pitches[max_magnitude_index]
    amplitude_perturbation = np.abs(np.diff(rms))
    peaks, _ = find_peaks(amplitude_perturbation)

    # Calculate shimmer features
    shimmer_apq3 = np.mean(np.abs(np.diff(amplitude_perturbation[peaks], 3)))/10
    shimmer_apq5 = np.mean(np.abs(np.diff(amplitude_perturbation[peaks], 5)))/10
    shimmer_dB = abs(np.log10(np.max(amplitude_perturbation) / rms.mean()))
    mdvp_shimmer = np.mean(amplitude_perturbation)

    return mdvp_shimmer, shimmer_dB, shimmer_apq3, shimmer_apq5

def calculate_shimmer_dda(audio_file_path):

    audio = AudioSegment.from_wav(audio_file_path)
    samples = np.array(audio.get_array_of_samples())
    shimmer = np.abs(samples[1:] - samples[:-1])
    shimmer_dda = np.sum(shimmer) / len(shimmer)
    
    return shimmer_dda/10000


def calculate_mdvp_apq(audio_file_path):
    # Load the audio file
    audio = AudioSegment.from_wav(audio_file_path)
    
    # Convert the audio to numpy array
    samples = np.array(audio.get_array_of_samples())
    
    # Calculate the time values
    time_values = np.linspace(0, len(samples) / audio.frame_rate, num=len(samples))
    
    # Perform linear regression to get the slope (F0 estimate)
    slope, _, _, _, _ = linregress(time_values, samples)
    
    # Calculate the standard deviation of the residuals (perturbations)
    residuals = samples - (slope * time_values)
    std_dev_residuals = np.std(residuals)
    
    # Calculate MDVP:APQ
    mdvp_apq = (std_dev_residuals / np.mean(np.abs(samples)))/100
    
    return mdvp_apq

def ppe(audio_file_path, frame_size=2048, hop_size=512):
    y, sr = librosa.load(audio_file_path)

    harmonic, percussive = librosa.effects.hpss(y)

    pitch, magnitudes = librosa.core.piptrack(y=harmonic, sr=sr)
    pitch = pitch[np.argmax(magnitudes, axis=0)]

    pitch[pitch == 0] = 1e-5

    pitch_period_freq = 1 / pitch
    mean_pitch_freq = np.nanmean(pitch_period_freq)
    return mean_pitch_freq/1000000
def preprocess_audio_params(wav_file_path):
    
    y, sr = librosa.load(wav_file_path)
    mdvp_shimmer, shimmer_dB, shimmer_apq3, shimmer_apq5 = calculate_shimmer_features(wav_file_path)
    features = {
        'MDVP:Fo(Hz)': abs(librosa.feature.mfcc(y=y, sr=sr)[0].mean()), #Mean fundamental frequency
        'MDVP:Fhi(Hz)': abs(librosa.feature.mfcc(y=y, sr=sr)[0].max()), #high
        'MDVP:Flo(Hz)': abs(librosa.feature.mfcc(y=y, sr=sr)[0].min()), #low
        'MDVP:Jitter(%)': calculate_mdvp_jitter(wav_file_path),#freq_variation
        'MDVP:Jitter(Abs)': calculate_mdvp_jitter_abs(wav_file_path,44100,4) ,
        'MDVP:PPQ': mdvp_ppq(wav_file_path),#over 5 pointsn
        'Jitter:DDP': (calculate_mdvp_jitter(wav_file_path)) * 3, 
        'MDVP:Shimmer':mdvp_shimmer, #variation in amplitude of consecutive speech cycles
        'MDVP:Shimmer(dB)': shimmer_dB,
        'Shimmer:APQ3': shimmer_apq3, #diff over 3
        'Shimmer:APQ5': shimmer_apq5,#diff over 5
        'MDVP:APQ': calculate_mdvp_apq(wav_file_path),
        'Shimmer:DDA': calculate_shimmer_dda(wav_file_path),
        'PPE':ppe(wav_file_path),

    }

    return features

# Example usage:
input_file_path = r"C:\Users\m0307\OneDrive\Desktop\dsa\datathon\991.m4a"  
output_file_path = r"C:\Users\m0307\OneDrive\Desktop\dsa\datathon\fileExp2.wav"  

convert_to_wav(input_file_path, output_file_path)
features_audio = preprocess_audio_params(output_file_path)
# df = pd.DataFrame(features_audio)

# # Display the DataFrame
# print(df)
for key, value in features_audio.items():
    if isinstance(value, Decimal):
        features_audio[key] = float(value)

# Convert the dictionary to a DataFrame
df = pd.DataFrame([features_audio])
df.to_csv("features.csv",index=False)
# Display the DataFrame
