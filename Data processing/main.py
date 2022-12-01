# This is a sample Python script.
import os

import librosa, librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from scipy.io import wavfile
import numpy as np
import constants


ravdess = "C:/Users/night/Desktop/Thesis/Data processing/ravdess/"
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 40


def process_ravdess_files():

    ravdess_directory_list = os.listdir(ravdess)

    file_emotion = []
    file_path = []


    for dir in ravdess_directory_list:
        directories = os.listdir(ravdess + dir)
        global count
        count = 0
        for file in directories:
            if count == 0:
                file_to_try = file
                file_to_try_path = ravdess + dir + '/' + file
            count += 1
            part = file.split('.')[0]
            part = part.split('-')[2]
            if part == "01" or part == "02":
                file_emotion.append("neutral")
            elif part == "03":
                file_emotion.append("happy")
            elif part == "04":
                file_emotion.append("sad")
            elif part == "05":
                file_emotion.append("angry")
            elif part == "06":
                file_emotion.append("fearful")
            elif part == "07":
                file_emotion.append("disgust")
            elif part == "08":
                file_emotion.append("surprised")

            file_path.append(ravdess + dir + '/' + file)


    # Create a dataframe for emotion of files
    emotion_data_frame = pd.DataFrame(file_emotion, columns=['Emotion'])

    # Create a dataframe for paths of files
    path_data_frame = pd.DataFrame(file_path, columns=['Path'])

    emotion_path_df = pd.concat([emotion_data_frame, path_data_frame], axis = 1)
    display(emotion_path_df.head(30))
    # display(emotion_data_frame)

    return file_to_try, file_to_try_path, file_path, file_emotion


def process_ravdess_files_as_raw_signals(file_path, file_emotion):

    raw_signals = []

    for i in range(0, len(file_path)):
        signal, sample_rate = librosa.load(file_path[i], sr=SAMPLE_RATE);
        raw_signals.append((signal, file_emotion[i]))

    return raw_signals

def process_ravdess_files_as_spectrums(raw_signals):

    fft_spectrums = []

    for signal in raw_signals:
        fft = np.fft.fft(signal[0])
        magnitude = np.abs(fft)
        frequency = np.linspace(0, SAMPLE_RATE, len(magnitude))
        left_frequency = frequency[:int(len(frequency) / 2)]
        left_magnitude = magnitude[:int(len(frequency) / 2)]
        fft_spectrums.append((((magnitude, frequency), (left_magnitude, left_frequency)), signal[1]))

    return fft_spectrums


def process_ravdess_files_as_spectrograms(raw_signals):

    spectograms_stft = []

    for signal in raw_signals:
        stft = librosa.core.stft(signal, hop_length = HOP_LENGTH, n_fft = N_FFT)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        spectograms_stft.append(((spectrogram, log_spectrogram), signal[1]))

    return spectograms_stft


def process_ravdess_files_as_mfccs(raw_signals):
    mfccs = []

    for signal in raw_signals:
        MFCCs = librosa.feature.mfcc(signal, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC)
        mfccs.append((MFCCs, signal[1]))


    return mfccs


def process_one_file(file_path, file):
    signal, sample_rate = librosa.load(file_path, sr = SAMPLE_RATE);

    # WAVEFORM

    # The signal is gonna be a numpy array
    # and will contain a number of values equal to sample_rate * duration of the sound
    # and for each of those values we will have the amplitude of the wave form

    librosa.display.waveshow(signal, sr = SAMPLE_RATE)
    plt.xlabel("Time")
    plt.ylabel("Amplitute")
    plt.show()

    # Like that we displayed the wave form for the loaded sound
    # Now we want to move from the time domain to the frequency domain and for
    # that we are going to apply FFT

    # FFT - FAST FOURIER TRANSFORM -> Spectrum

    fft = np.fft.fft(signal)
    # We expect to obtain a one dimensional array which has as many values
    # as the total number of samples that were in the waveform -> sr * T
    # and for each of these values we have a complex value
    # Now we want to move from that complex value and get the magnitude for those values

    magnitude = np.abs(fft)
    # We apply the absolute value on those complex values and then we end up with the magnitudes
    # And these magnitudes indicate the contribution of each frequency bin to the overall sound
    # And we want to map them onto the relative frequency bins

    frequency = np.linspace(0, SAMPLE_RATE, len(magnitude))
    # The function linspace gives us a number of evenly spaced numbers in an interval
    # And here the frequency interval that we want to consider is between 0Hz and the sample rate itself.
    # And the number of evenly spaced numbers from the interval that we want to get is equal to the length
    # Of the magnitude

    # The arrays magnitude and frequency tells us how much each frequency is contributing to the overall sound

    plt.plot(frequency, magnitude)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

    # So now we obtained our power spectrum
    # As an observation, we can see that the plot we obtianed is symmetrical
    # And we don't need the whole plot, we only need the first half of the plot
    # Because it gives us novel information and that's because when we cross half the frequency
    # It is just repeating the same information symmetrically.

    left_frequency = frequency[:int(len(frequency)/2)]
    left_magnitude = magnitude[:int(len(frequency)/2)]

    plt.plot(left_frequency, left_magnitude)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

    # Now we want to understand how those frequencies are contributing to the overall sound throughout time

    # Short Time Fourier Transform (STFT) -> Spectrogram
    # The spectogram is going to give us information about magnitude as a function of both frequency and time

    #Number of samples / fft
    n_fft = 2048 # number of samples
    hop_length = 512 # this represents the amounts we are shifting each fourier transform to the right

    # When we do a FFT we slide like an interval and at each interval we calculate a FFT and
    # the hop tells us how much we are sliding to the right

    stft = librosa.core.stft(signal, hop_length = hop_length, n_fft = n_fft)

    spectrogram = np.abs(stft)

    librosa.display.specshow(spectrogram, sr = SAMPLE_RATE, hop_length = hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.show()

    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    librosa.display.specshow(log_spectrogram, sr = SAMPLE_RATE, hop_length = hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.show()

    # Apply MFCCs
    # n_mfcc -> the number of coefficients we want to extract

    MFCCs = librosa.feature.mfcc(signal, n_fft = n_fft, hop_length = hop_length, n_mfcc = 13)
    librosa.display.specshow(MFCCs, sr = SAMPLE_RATE, hop_length = hop_length)
    plt.xlabel("Time")
    plt.ylabel("MFCC")
    plt.colorbar()
    plt.show()


if __name__ == '__main__':

    file_to_try, file_to_try_path, file_path = process_ravdess_files()

    print(file_to_try_path)
    print(file_to_try)
    process_one_file(file_to_try_path, file_to_try)
    # process_ravdess_files_as_raw_signals(file_path)

