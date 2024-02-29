import warnings
from joblib import Parallel, delayed
import numpy as np
from pipeline import Pipeline
from preprocessing_library import FFT, Slice, Magnitude, Log10
import collections
import dill as pickle
import argparse
import platform
import os
import sys
from matplotlib import pyplot as plt

print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0, os.getcwd())


seizure_type_data = collections.namedtuple(
    'seizure_type_data', ['patient_id', 'seizure_type', 'data'])


def convert_to_fft(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency, file_path):

    warnings.filterwarnings("ignore")
    type_data = pickle.load(open(file_path, 'rb'))
    pipeline = Pipeline(
        [FFT(), Slice(fft_min_freq, fft_max_freq), Magnitude(), Log10()])
    time_series_data = type_data.data
    start, step = 0, int(np.floor(window_step * sampling_frequency))
    stop = start + int(np.floor(window_length * sampling_frequency))
    fft_data = []

    while stop < time_series_data.shape[1]:
        signal_window = time_series_data[:][start:stop]
        fft_window = pipeline.apply(signal_window)
        fft_data.append(fft_window)
        start, stop = start + step, stop + step

    fft_data = np.array(fft_data)
    # named_data = seizure_type_data(
    #     patient_id=type_data.patient_id, seizure_type=type_data.seizure_type, data=fft_data)

    # return named_data, os.path.basename(file_path)
    return fft_data


save_data_dir = "E:/new_data/train/bckg/aaaaaaac_s004_t000_0.pkl"

# fnames = []
# for (dirpath, dirnames, filenames) in os.walk(save_data_dir):
#     fnames.extend(filenames)

# fpaths = [os.path.join(save_data_dir,f) for f in fnames]

sampling_frequency = 250  # Hz
fft_min_freq = 1  # Hz

window_lengths = [1, 2, 4, 8, 16]#[0.25, 0.5, 1]#[1, 2, 4, 8, 16]
fft_max_freqs = [12, 24, 48, 64, 96]#[12, 24]

for window_length in window_lengths:
    window_steps = list(np.arange(window_length/4, window_length/2 + window_length/4, window_length/4))

    print(window_steps)
    for window_step in window_steps:
        for fft_max_freq_actual in fft_max_freqs:
            fft_max_freq = fft_max_freq_actual * window_length
            fft_max_freq = int(np.floor(fft_max_freq))
            print('window length: ', window_length, 'window step: ', window_step, 'fft_max_freq', fft_max_freq)
            # save_data_dir = os.path.join(preprocess_data_dir, 'fft_seizures_' + 'wl' + str(window_length) + '_ws_' + str(window_step) \
            #                 + '_sf_' + str(sampling_frequency) + '_fft_min_' + str(fft_min_freq) + '_fft_max_' + \
            #                 str(fft_max_freq_actual))
            # if not os.path.exists(save_data_dir):
            #     os.makedirs(save_data_dir)
            # else:
            #     exit('Pre-processed data already exists!')


            converted_data = convert_to_fft(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency,save_data_dir)

            plt.imshow(converted_data, interpolation='nearest')
            plt.show()
            