import os
import sys
import platform
import argparse
import pandas as pd
import numpy as np
import math
import collections
from tabulate import tabulate
import pyedflib
import re
from scipy.signal import resample
import pickle
import h5py
import progressbar
from time import sleep
if len(sys.argv) != 3:
    print("Usage: python script.py input_data_directory save_data_dir")
    sys.exit(1)

parameters = pd.read_csv('data_preparation/parameters.csv', index_col=['parameter'])

def extract_signal(f, signal_labels, electrode_name, start, stop):

    tuh_label = [s for s in signal_labels if 'EEG ' + electrode_name + '-' in s]

    if len(tuh_label) > 1:
        print(tuh_label)
        exit('Multiple electrodes found with the same string! Abort')

    channel = signal_labels.index(tuh_label[0])
    signal = np.array(f.readSignal(channel))

    start, stop = float(start), float(stop)
    original_sample_frequency = f.getSampleFrequency(channel)
    original_start_index = int(np.floor(start * float(original_sample_frequency)))
    original_stop_index = int(np.floor(stop * float(original_sample_frequency)))

    seizure_signal = signal[original_start_index:original_stop_index]

    new_sample_frequency = int(parameters.loc['sampling_frequency']['value'])
    new_num_time_points = int(np.floor((stop - start) * new_sample_frequency))
    seizure_signal_resampled = resample(seizure_signal, new_num_time_points)

    return seizure_signal_resampled

def read_edfs_and_extract(edf_path, edf_start, edf_stop):

    f = pyedflib.EdfReader(edf_path)

    montage = str(parameters.loc['montage']['value'])
    montage_list = re.split(';', montage)
    signal_labels = f.getSignalLabels()
    x_data = []

    for i in montage_list:
        electrode_list = re.split('-', i)
        electrode_1 = electrode_list[0]
        extracted_signal_from_electrode_1 = extract_signal(f, signal_labels, electrode_name=electrode_1, start=edf_start, stop=edf_stop)
        electrode_2 = electrode_list[1]
        extracted_signal_from_electrode_2 = extract_signal(f, signal_labels, electrode_name=electrode_2, start=edf_start, stop=edf_stop)
        this_differential_output = extracted_signal_from_electrode_1-extracted_signal_from_electrode_2
        x_data.append(this_differential_output)

    f._close()
    del f

    x_data = np.array(x_data)
    return x_data


input_dir = sys.argv[1] # Input path
output_dir = sys.argv[2]  # Output Path

classes = ['seiz', 'bckg']
# make folders in new_data
for set_folder in ['train', 'dev', 'eval']:
    for class_label in classes:
        os.makedirs(os.path.join(output_dir, set_folder, class_label), exist_ok=True)

# Iterate through each set folder (train, dev, eval)
for set_folder in ['train', 'dev', 'eval'][:1]:
    set_path = os.path.join(input_dir, set_folder)
    # Iterate through each patient folder
    for patient_folder in os.listdir(set_path)[:1]:
        # print(set_folder, patient_folder)
        patient_path = os.path.join(set_path, patient_folder)

        # Iterate through each sub_folder
        try:
            for folder in os.listdir(patient_path):
                # print(folder)
                file_path = os.path.join(patient_path, folder)

                for sub_folder in os.listdir(file_path):
                    # print()
                    main_path = os.path.join(file_path, sub_folder)

                    for csv_file in os.listdir(main_path):
                        if csv_file.endswith('.csv_bi'):

                            csv_path = os.path.join(main_path, csv_file)
                            csv_name = csv_file.split(".")[0]

                            df = pd.read_csv(csv_path, skiprows=5)
                            
                            count = 0
                            for index, row in df.iterrows():
                                start_time = row['start_time']
                                end_time = row['stop_time']
                                seizure_class = row['label']
                                print(start_time, type(start_time))
                                print(end_time, type(end_time))
                                print(seizure_class)
                                edf_path = os.path.join(main_path, csv_name + '.edf')

                                data = read_edfs_and_extract(edf_path, start_time, end_time)
                                print(edf_path)
                                
                                with open((output_dir + set_folder + '//' + seizure_class + '//' + csv_name + '_' + str(count) + '.pkl'), 'wb') as myfile:
                                    pickle.dump(data, myfile)
                                
                                count+=1
        except Exception as e:
            print(e)
