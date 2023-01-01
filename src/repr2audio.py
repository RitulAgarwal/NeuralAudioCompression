from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import BatchNormalization,LeakyReLU
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error,mean_squared_error
import pandas as pd
from matplotlib import pyplot as plt
import logging
import keras
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import matplotlib.pyplot as plt
from src.train_compression_model import train_compression_model
from scipy.io import wavfile


scaler = MinMaxScaler(feature_range = (0,1))

class Repr2Audio:
    def __init__(self,test_data):
        self.test_data = test_data

    def test_data_prep(self):
        data = pd.read_csv(self.test_data, header=None, index_col=None)
        batch_X = data.values.astype(np.float32)
        return batch_X,batch_X

    def log_melspec2audio(self):
        pass
    
    #def testing(self):  
    #    checkpoint_path = "/home/avpl/Documents/audio_related/OOPS_program/checkpoints/174-0.000880.hdf5"
    #    tcm = train_compression_model('recorded_voice.wav',256)
    #    model = tcm.my_model()
    #    model.load_weights(checkpoint_path)
    #    data = scaler.inverse_transform(scaled_mfccs)
    #    data_actual = librosa.feature.inverse.mfcc_to_audio(data)
    #    file_name = "final.wav"
    #    wavfile.write(file_name,20000, data_actual)
            
            