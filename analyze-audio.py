#!/usr/bin/python3

import config
import soundfile as sf
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import librosa
from glob import glob

def main():

    # load sample file
    sample_file = 'samples/sample-01-raw.wav'
    data_points, sr = librosa.load(sample_file)

    # get meta data
    data_shape = data_points.shape
    amount_of_chunks = math.floor(data_shape[0] / config.BUFFER_SIZE )

    # print meta data
    print(f'Loading file: {sample_file}')
    print(f'example data_points: {data_points[:10]}') 
    print(f'shape data_points: {data_points.shape[0]}')
    print(f'amount of chunks: {amount_of_chunks}')

    # show the raw file data
    pd.Series(data_points).plot(
        figsize=(10, 5),
        lw=1,
        title=f'Raw Data for {sample_file}',
        color=config.COLOR_PAL[0]
    )
    plt.show()

    # convert the data in our format
    chunks = np.array_split(data_points, amount_of_chunks)
    
    # gets the sum of each chunk and add/remove the
    # error margin to its sum
    chunks_data = []
    x_axis = []
    i = 0
    for chunk in chunks:
        chunk = np.abs(chunk)
        chunk_sum = np.sum(chunk)
        chunks_data.append(chunk_sum)
        x_axis.append(i)
        i += 1

    # display the data
    plt.scatter(x_axis, chunks_data)
    plt.show()

    # display the data
    plt.scatter(x_axis[:10], chunks_data[:10])
    plt.show()

    # dispplay data with error_margin
    plt.errorbar(
        x_axis[:10],
        chunks_data[:10],
        yerr=config.ERROR_MARGIN,
        fmt="o",
        capsize=10
    )
    plt.show()

    # show the frequency domain of the first 256 ticks
    ticks = 256 
    D = np.abs(librosa.stft(data_points[:ticks], n_fft=ticks, hop_length=ticks+1))
    plt.plot(D)
    plt.show()

    # display spectogram
    data_points_db = librosa.stft(data_points)
    data_points_db_flat = librosa.amplitude_to_db(np.abs(data_points_db), ref=np.max)
    plt.figure()
    librosa.display.specshow(data_points_db_flat)
    plt.colorbar()
    plt.show()

    # show the mel spectogram
    fig, ax = plt.subplots()
    M = librosa.feature.melspectrogram(y=data_points, sr=sr)
    M_db = librosa.power_to_db(M, ref=np.max)
    img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax)
    ax.set(title='Mel spectrogram display')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()

if __name__ == '__main__':
    main()

