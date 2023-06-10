import matplotlib.pyplot as plt
import numpy as np 
import librosa
import librosa.display
import sys

def convert_wav_to_spectrogram():
    temp_file_path = sys.argv[1]
    spectrogram_path = sys.argv[2]

    y, sr = librosa.load(temp_file_path)
    librosa.feature.melspectrogram(y=y, sr=sr)
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000, ax=ax)
    #plt.imshow(img, aspect='auto', origin='lower')

    # # Simpan spektrogram sebagai gambar PNG
    plt.savefig(spectrogram_path)
    plt.close()
    return {'spectrogramPath': spectrogram_path}

convert_wav_to_spectrogram()
