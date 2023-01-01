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
from hyperopt import hp
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
    def __init__(self,file_path,batch_size):
        self.file_path = file_path
        self.batch_size = batch_size
        self.epochs = 200
        self.train_gen, self.valid_gen = self.get_datagens()
        

        self.sr = 22050
        self.optim_adam=keras.optimizers.Adam(
            learning_rate=0.009,
            name='Adam',
        )
        self.steps_per_epoch = 50
        self.validation_steps = 10

        self.list_of_all_audios = os.listdir(self.file_path)
        self.num_audio = len(self.list_of_all_audios)
        self.checkpoint_path = "checkpoints_mel/{epoch:02d}-{val_loss:.6f}.hdf5"
        self.model_checkpoint_callback = ModelCheckpoint(self.checkpoint_path,
                            monitor='val_loss', mode="min",
                            verbose=1, save_best_only=True, save_weights_only=False, save_freq="epoch")
        self.earlystop_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, 
                                                patience=30)
        self.lr_schedule = ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=5,
                                                verbose=1, mode="min")

    def datagen(self,mode):
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
            audio_batch = list(random.sample(relevant_data,self.batch_size))
            batch_X=[]
            for i,audio in enumerate(audio_batch):
                scale,sr = librosa.load(os.path.join('/home/avpl/Documents/audio_related/OOPS_proj/OneSecAudios',audio))
                mel_spectrogram = librosa.feature.melspectrogram(y=scale,sr=sr,n_mels=130)#TODO n_fft and hop_length to be added
                img = (mel_spectrogram * 255).astype(np.uint8)
                #img.load_image(image)
                #print(np.shape(img))
                # img.save(melspec_path +str(i)+ "_"+((audio).split("OneSecAudios/")[1]) )
                batch_X.append(img)
            batch_X = np.array(batch_X)
            yield tuple((batch_X, batch_X))

                                                                     
    def get_datagens(self):
        train_gen = self.datagen(mode='train')
        valid_gen = self.datagen(mode='valid')
        return train_gen, valid_gen

    def objective(self,config):
           # 1. Wrap a Keras model in an objective function.
        autoencoder = self.my_model()
        autoencoder.compile(optimizer="adam", loss= "mae")
        history = autoencoder.fit(
                            x = self.train_gen,
                            validation_data=self.valid_gen,
                            epochs=config["epochs"],
                            steps_per_epoch=self.steps_per_epoch,
                            batch_size=self.batch_size,
                            validation_steps=self.validation_steps,
                            verbose=1
                            #callbacks=callbacks
                            )
        return {"mae": max(history.history['val_loss'])}



    def ray_tune(self):
        search_space = {
	        "lr": tune.choice("lr", [0.0001, 0.001, 0.01, 0.1]),
	        "batch_size": tune.choice("batch_size", [8, 16, 32, 64]), 

        }
        algo = HyperOptSearch()

        # 3. Start a Tune run that maximizes accuracy.
        tuner = tune.Tuner(
            self.objective,
            tune_config=tune.TuneConfig(
                metric="mae",
                mode="min",
                search_alg=algo,
            ),
            param_space=search_space,
        )
        results = tuner.fit()      

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

    
    def train_model(self):
        autoencoder = self.my_model()
        autoencoder.compile(self.optim_adam, loss= "mae",metrics=['accuracy'])
        callbacks = [
            self.model_checkpoint_callback,
            #self.earlystop_callback,
            self.lr_schedule
        ]
        history = autoencoder.fit(
                            x = self.train_gen,
                            validation_data=self.valid_gen,
                            epochs=self.epochs,
                            steps_per_epoch=self.steps_per_epoch,
                            batch_size=self.batch_size,
                            validation_steps=self.validation_steps,

                            verbose=1
                            #callbacks=callbacks
                            )
        
        print("done till model compile")
        #plt.plot(history.history['mae'])
        #plt.plot(history.history['val_mae'])
        #plt.title('model mae')
        #plt.ylabel('mae')
        #plt.xlabel('epoch')
        #plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        ##
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
            
    # def test_data_prep(self):
    #    data = pd.read_csv(self.data, header=None, index_col=None)
    #    batch_X = data.values.astype(np.float32)
    #    return batch_X,batch_X
        
##CUSTOM LOSS
#LR CHANGE
# VALIDATION STEPS_PER_EPOCH
#EPOCHS
    #ONLY FOR RYTUNE WALI CHEEZ ]\
    def train_model(self):
        autoencoder = self.my_model()
        autoencoder.compile(self.optim_adam, loss= "mae",metrics=['accuracy'])
        callbacks = [
            self.model_checkpoint_callback,
            #self.earlystop_callback,
            self.lr_schedule
        ]
        history = autoencoder.fit(
                            x = self.train_gen,
                            validation_data=self.valid_gen,
                            epochs=self.epochs,
                            steps_per_epoch=self.steps_per_epoch,
                            batch_size=self.batch_size,
                            validation_steps=self.validation_steps,

                            verbose=1
                            #callbacks=callbacks
                            )
        
        print("done till model compile")
        #plt.plot(history.history['mae'])
        #plt.plot(history.history['val_mae'])
        #plt.title('model mae')
        #plt.ylabel('mae')
        #plt.xlabel('epoch')
        #plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        ##
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


    




