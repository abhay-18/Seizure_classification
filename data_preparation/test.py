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

def convert_to_fft(window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency, time_series_data):

    warnings.filterwarnings("ignore")
    # print("H3", type_data.shape)
    pipeline = Pipeline(
        [FFT(), Slice(fft_min_freq, fft_max_freq), Magnitude(), Log10()])
    # time_series_data = np.array(type_data.data)
    print(time_series_data.shape)
    start, step = 0, int(np.floor(window_step * sampling_frequency))
    stop = start + int(np.floor(window_length * sampling_frequency))
    fft_data = []
    # print("H5", start, stop, step)
    while stop <= time_series_data.shape[1]:
        # print(stop, time_series_data.shape[1])
        # print("H6",start, stop)
        signal_window = time_series_data[:, start:stop]
        # print("H2" ,signal_window.shape)
        fft_window = pipeline.apply(signal_window)
        fft_data.append(fft_window)
        start, stop = start + step, stop + step

    fft_data = np.array(fft_data)
    return fft_data


input_data_dir = "E:/new_data/train/bckg/aaaaabom_s005_t005_0.pkl"
# preprocessed_data_dir = ""

# classes = ['bckg']
# make folders in new_data
# for set_folder in ['train', 'dev', 'eval']:
#     for class_label in classes:
#         os.makedirs(os.path.join(preprocessed_data_dir,
#                     set_folder, class_label), exist_ok=True)


sampling_frequency = 250  # Hz
fft_min_freq = 1  # Hz

window_lengths = [1]
fft_max_freqs = [126]

# for set_folder in ['train', 'dev', 'eval']:
#     set_path = os.path.join(input_data_dir, set_folder)

#     for label in classes:
#         main_path = os.path.join(set_path, label)

#         try:
#             for files in os.listdir(main_path):
#                 sample_path = os.path.join(main_path, files)
#                 file_name = files.split('.')[0]
#                 print(sample_path)
for window_length in window_lengths:
    # window_steps = list(np.arange(window_length/4, window_length/2 + window_length/4, window_length/4))
    window_steps = [0.25]
    for window_step in window_steps:
        for fft_max_freq_actual in fft_max_freqs:
            fft_max_freq = fft_max_freq_actual * window_length
            fft_max_freq = int(np.floor(fft_max_freq))
            print('window length: ', window_length, 'window step: ',
                  window_step, 'fft_max_freq', fft_max_freq)

            type_data = pickle.load(open(input_data_dir, 'rb'))
            time_series_data = np.array(type_data.data)

            if(time_series_data.shape[1] < 250):
                continue
            else:

                converted_data = convert_to_fft(
                    window_length, window_step, fft_min_freq, fft_max_freq, sampling_frequency, time_series_data)
                # print(converted_data.shape)
                converted_data = np.reshape(
                    converted_data, (converted_data.shape[1], converted_data.shape[2], converted_data.shape[0]))
                print(converted_data.shape)
                with open(('./test.pkl'), 'wb') as myfile:
                    pickle.dump(converted_data, myfile)

        # except Exception as e:
        #     print(e)
