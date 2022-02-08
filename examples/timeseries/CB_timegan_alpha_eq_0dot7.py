#Importing the required libs for the exercise
#Importing the required libs for the exercise
import os
path = 'Z:/faellesmappe/cmd/tfs_alt/ydata-synthetic/src'
os.chdir(path)
print("Current working directory: {0}".format(os.getcwd()))
import sys
sys.path.append(path)

from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)
    
import pickle
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeGAN
from sklearn.model_selection import TimeSeriesSplit
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TF to use only the CPU
tf.config.set_visible_devices([], 'GPU')

def fit_ar_1(ts):
    try:
        x = ts[0:-1]
        y = ts[1:]
        p = np.polyfit(x.reshape(-1), y.reshape(-1), 1)
        beta = p[0]

        c = np.mean(ts) * (1 - beta)
        sigma = 0;
    except:
        beta = 0;
        c = 0;
        sigma = 0;

    return beta, c, sigma


def simulate_ar_1(alpha, T, warm = 1000):
    innovations = np.random.normal(size = T + warm);

    # this is retarded, find out how to use a filter instead
    y = [];
    for i in range(len(innovations)):
        if i == 0:
            y.append(innovations[i]);
            continue;

        y.append(y[i - 1] * alpha + innovations[i]);

    return y[warm:];



# Method implemented here: https://github.com/jsyoon0823/TimeGAN/blob/master/data_loading.py
# Originally used in TimeGAN research
def real_data_loading(data: np.array, seq_len):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: Numpy array with the values from a a Dataset
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """
    # Flip the data to make chronological data
    ori_data = data[::-1]
    # Normalize the data
    scaler = MinMaxScaler().fit(ori_data)
    ori_data = scaler.transform(ori_data)

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    return data

def CB_timeGAN(synth,alpha_true, T=1000,warm = 1000,seq_len=150,replications=1000,resamples=10000,
               folder='W:/BDADSharedData/CBGAN/timeGANSimulations/'):
    
    estimates_all_replications = []
    for capM in tqdm(range(replications)):
        y  = np.array(simulate_ar_1(alpha_true, T, warm)).reshape(-1,1)
        y  = real_data_loading(y,seq_len)
        
        synth = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=seq_len, n_seq=n_seq, gamma=1)
        synth.train(y, train_steps=1000)
        synth_data = synth.sample(resamples)
        
        estimates = [];
        for i in range(synth_data.shape[0]):
            sample = synth_data[i, :];

            try:
                alpha, _, _ = fit_ar_1(sample);
                estimates.append(alpha);
            except np.linalg.LinAlgError:
                print('warning: ar1 fit convergence error.');
                alpha, _, _ = (0, 0, 0);
        estimates_all_replications.append(estimates)
        with open(folder + f'alpha_estimation_all_replications_{alpha_true}_{seq_len}.pkl', 'wb') as f:
            pickle.dump(estimates_all_replications, f)

    

#Specific to TimeGANs
seq_len=150
n_seq = 1
hidden_dim=24
gamma=1
noise_dim = 32
dim = 128
batch_size = 32
log_step = 100
learning_rate = 5e-4
gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           noise_dim=noise_dim,
                           layers_dim=dim)

CB_timeGAN(gan_args,alpha_true=0.7)
