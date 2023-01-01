from keras.models import Sequential 
from keras.layers import Activation, Dense,Input
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
logging.basicConfig(
    format='[%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

class train_mfcc_compression_model:
    def __init__(self,data,batch_size):
        data = pd.read_csv(data, header=None, index_col=None).sample(frac = 1) # shuffled
        self.data = data
        self.batch_size = batch_size
        self.optim_adam=keras.optimizers.Adam(
            learning_rate=0.01,
            name='Adam',
        )
        # TODO ---empty checkpoint dir at eery run
       #self.checkpoint_path = "checkpoints/{epoch:02d}-{val_loss:.6f}.hdf5"
       #self.model_checkpoint_callback = ModelCheckpoint(self.checkpoint_path,
       #                    monitor='val_loss', mode="min",
       #                    verbose=0, save_best_only=True, save_weights_only=False, save_freq="epoch")
       #self.earlystop_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=0, 
       #                                        patience=100)# min_delta=0.001 )
       #self.lr_schedule = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=30,
       #                                        verbose=0, mode="min")
       #self.epochs = 500
        self.train_gen, self.valid_gen = self.get_datagens()

    def datagen(self,mode):
        self.mode = mode
        train_val_split = 0.8
        while True:
            if mode == "train":
                relevant_data = self.data[:int(len(self.data) * train_val_split)]
            elif mode == "valid":
                relevant_data = self.data[int(len(self.data) * train_val_split):]            
            #print(type(relevant_data))
            batch_X = (relevant_data.sample(n = self.batch_size)).values.astype(np.float32)
            #batch_shape=print(batch_X.shape)
            yield batch_X, batch_X
                                                                                         
    def get_datagens(self):
        train_gen = self.datagen(mode = 'train')
        valid_gen = self.datagen(mode='valid')
        return train_gen, valid_gen
          
    def my_model(self):
        n_inputs =39
        visible = Input(shape=(n_inputs,))
        # encoder level 1
        #e = Dense(33)(visible)
        #e = LeakyReLU()(e)
        # encoder level 2
        e = Dense(28)(visible)
        e = LeakyReLU()(e)
        #encoder level 3
        e = Dense(20)(e)
        e = LeakyReLU()(e)
        #encoder level 4
        e = Dense(15)(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        #encoder level 5
        e = Dense(10)(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        # bottleneck
        n_bottleneck = 5
        bottleneck = Dense(n_bottleneck)(e)
        # define decoder, level 1
        d = Dense(10)(bottleneck)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        # decoder level 2
        d = Dense(15)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        #decoder level 3
        d = Dense(20)(d)
        d = LeakyReLU()(d)
        #decoder level 4
        d = Dense(28)(d)
        d = LeakyReLU()(d)
        #decoder level 5
        #d = Dense(33)(d)
        #d = LeakyReLU()(d)
        ## output layer
        output = Dense(n_inputs)(d)
        # define autoencoder model
        model = keras.Model(inputs=visible, outputs=output)
        return model

    def train_model(self):
        
        model = self.my_model()
        model.compile(self.optim_adam, loss='mse', metrics=["mae"])
        callbacks = [
            self.model_checkpoint_callback,
            self.earlystop_callback,
            self.lr_schedule
        ]
        history = model.fit(self.train_gen,
                            epochs=self.epochs,
                            steps_per_epoch=self.steps_per_epoch,
                            batch_size=self.batch_size,
                            validation_data=self.valid_gen,
                            validation_steps=self.validation_steps,
                            verbose=1,
                            callbacks=callbacks)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('model mae')
        plt.ylabel('mae')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
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