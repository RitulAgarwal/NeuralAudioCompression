from keras.models import Sequential 
from keras.layers import Activation, Dense, Input, BatchNormalization, LeakyReLU, Conv2D, MaxPooling2D, Conv2DTranspose,UpSampling2D,Cropping2D
import keras as keras
import math,os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error,mean_squared_error
import pandas as pd
from keras.models import Model
from matplotlib import pyplot as plt
import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import matplotlib.pyplot as plt
import librosa
import random
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
import keras
from PIL import Image
logging.basicConfig(
    format='[%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)


def custom_loss(y_pred, y_true):
    return keras.reduceSum(y_pred)

class train_melspec_compression_model:
    def __init__(self,file_path,config={}):
        self.file_path = file_path
        self.batch_size = config['batch_size']
        self.lr = config['lr']
        self.config = config

        self.epochs = 500

        self.sr = 22050
        self.steps_per_epoch = 50
        self.validation_steps = 10
        self.list_of_all_audios = os.listdir(self.file_path)
        self.num_audio = len(self.list_of_all_audios)

        self.checkpoint_path = "checkpoints_mel/{epoch:02d}-{val_loss:.6f}.hdf5"
        self.model_checkpoint_callback = ModelCheckpoint(self.checkpoint_path,
                            monitor='val_loss', mode="min",
                            verbose=1, save_best_only=True, save_weights_only=False, save_freq="epoch")
        self.earlystop_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, 
                                                patience=16)
        self.lr_schedule = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                                verbose=1, mode="min")

    def datagen(self,mode, bs):
        #melspec_path = 'res/log_melspecs/'
        #os.makedirs(melspec_path, exist_ok=True)
        self.mode = mode
        train_val_split = 0.8
        while True:
            if mode == "train":
                relevant_data = self.list_of_all_audios[:math.floor(self.num_audio * train_val_split)]
            elif mode == "valid":
                relevant_data = self.list_of_all_audios[math.floor(self.num_audio * train_val_split):]    
            #audio_batch = list((relevant_data.sample(n = self.batch_size))["Filenames"])
            audio_batch = list(random.sample(relevant_data,bs))
            batch_X=[]
            for i,audio in enumerate(audio_batch):
                scale,sr = librosa.load(os.path.join('/home/avpl/Documents/audio_related/OneSecAudios',audio))
                mel_spectrogram = librosa.feature.melspectrogram(y=scale,sr=sr,n_mels=130)#TODO n_fft and hop_length to be added
                img = (mel_spectrogram * 255).astype(np.uint8)
                #new_image = Image.fromarray(img)
                #new_image.save('new.png')   
                #img.load_image(image)
                print(np.shape(img))
                # img.save(melspec_path +str(i)+ "_"+((audio).split("OneSecAudios/")[1]) )
                batch_X.append(img)
            batch_X = np.array(batch_X)
            yield tuple((batch_X, batch_X))

                                                                     
    def get_datagens(self, batch_size):
        train_gen = self.datagen(mode='train', bs=batch_size)
        valid_gen = self.datagen(mode='valid', bs=batch_size)
        return train_gen, valid_gen


    def my_model(self):
        input = Input(shape=(130,13, 1),name = "input")
        x = Conv2D(32, (16,2), activation='relu', padding='valid')(input)
        x = Conv2D(16, (9,1), activation='relu', padding='valid')(x)
        x = MaxPooling2D(pool_size=(4,1))(x)
        x = Conv2D(16, (10,2), activation='relu', padding='valid')(x)
        x = Conv2D(8, (9,9), activation='relu', padding='valid')(x)
        x = Conv2DTranspose(16, (10, 5), activation="relu", padding="valid")(x)
        x = Conv2DTranspose(32, (10, 1), activation="relu", padding="valid")(x)
        x = UpSampling2D((5,2))(x)
        x = Cropping2D(cropping = ((4,1),(1,0)))(x)
        
        x = Conv2D(1, (3, 3), activation="relu", padding="same")(x)

        autoencoder = Model(input, x)
        return autoencoder

    
    def train_model(self, bs=16, lr=0.0003410346690618139):
        autoencoder = self.my_model()
        optim_adam=keras.optimizers.Adam(
            learning_rate=lr, name='Adam')
        autoencoder.compile(optim_adam, loss= "mae")
        callbacks = [
            self.model_checkpoint_callback,
            self.earlystop_callback,
            self.lr_schedule    
        ]

        train_gen, valid_gen = self.get_datagens(bs)

        history = autoencoder.fit(
                            x = train_gen,
                            validation_data=valid_gen,
                            epochs=self.epochs,
                            steps_per_epoch=self.steps_per_epoch,
                            batch_size=bs,
                            validation_steps=self.validation_steps,
                            verbose=0,
                            callbacks=callbacks
                            )
        print("done till model compile")
      
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        return history