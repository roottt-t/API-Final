from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

########################
# Load the example clip.
y, sr = librosa.load('mixture/mixture.wav', offset=40, duration=10)


###############################################
# Compute the short-time Fourier transform of y
D = librosa.stft(y)


y1, sr = librosa.load('mixture/vocals.wav', offset=40, duration=10)


###############################################
# Compute the short-time Fourier transform of y
D_vocal = librosa.stft(y1)


y2, sr = librosa.load('mixture/bass.wav', offset=40, duration=10)


###############################################
# Compute the short-time Fourier transform of y
D_bass = librosa.stft(y2)


y3, sr = librosa.load('mixture/drums.wav', offset=40, duration=10)


###############################################
# Compute the short-time Fourier transform of y
D_drum= librosa.stft(y3)


y4, sr = librosa.load('mixture/other.wav', offset=40, duration=10)


###############################################
# Compute the short-time Fourier transform of y
D_other = librosa.stft(y4)


####################################################################
# We can plot the two components along with the original spectrogram

# Pre-compute a global reference power from the input spectrum
rp = np.max(np.abs(D))

plt.figure(figsize=(12, 10))

plt.subplot(5, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=rp), y_axis='log')
plt.colorbar()
plt.title('Full spectrogram')

plt.subplot(5, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_vocal), ref=rp), y_axis='log')
plt.colorbar()
plt.title('Vocal spectrogram')

plt.subplot(5, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_bass), ref=rp), y_axis='log', x_axis='time')
plt.colorbar()
plt.title('Bass spectrogram')

plt.subplot(5, 1, 4)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_drum), ref=rp), y_axis='log', x_axis='time')
plt.colorbar()
plt.title('Drum spectrogram')

plt.subplot(5, 1, 5)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_other), ref=rp), y_axis='log', x_axis='time')
plt.colorbar()
plt.title('Other spectrogram')

plt.tight_layout()
plt.show()