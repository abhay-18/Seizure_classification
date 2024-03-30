# How to prepare data

Only support Python 3.

# Step 0: Download TUH data
Please download from [https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) 

## Step 1: Build Data
Raw seizure dataset contains seizure and no seizure data in different sampling rate. 
'build_data_new.py' is to get the seizure data with different seizure types from raw dataset specified by 'input_data_dir' and 
then use a common sampling rate (by default 250Hz) and save to 'save_data_dir'

Exemplary usage:                        
```
python3 data_preparation/build_data.py --input_data_dir [Path to Raw data]  
    --save_data_dir [Path to seizure type data] 
```

## Step 2 Preprocess the data
We have preprocessed the data to generate fft images.

Exemplary usage:                        
```
python3 data_preparation/generate_fft_images.py --input_data_dir [Path to seizure type data] 
    --preprocess_data_dir [Path to preprocessed seizure type data]
```

## Step 3 Generate cross validation folds

Exemplary usage                    
```
python3 data_preparation/generate_cross_val_folds.py 
    --save_data_dir  [Path to seizure type data] --tuh_eeg_szr_ver [v1.5.2 or v1.4.0]
```
It will out put 3 pkl file which is already been pre-generated.
```
cv_split_5_fold_seizure_wise_v1.4.0.pkl
cv_split_5_fold_seizure_wise_v1.5.2.pkl
cv_split_3_fold_patient_wise_v1.5.2.pkl
```
Please note v1.4.0 doesn't support patient wise cross validation because some seizure types only have 2 patients. 
You will expect lower performance for patient wise cross validation because the testing patients' data are not in the training.
