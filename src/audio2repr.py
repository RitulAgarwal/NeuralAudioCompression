import csv
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import librosa.display
import logging
import matplotlib.pyplot as plt
logging.basicConfig(
    format='[%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
from PIL import Image


class Audio2Repr:
    def __init__(self, audio_files, sr,name):
        self.audio_files = audio_files
        self.sr = 22050
        self.name = name
        self.chunked_audio_dir = '1_sec_audios/'
    

    def melspec_extractor(self):
        #FOR 1 SEC AUDIO(130,44) ---- (number of Mel bins,the number of features)
        melspec_path = 'res/log_melspecs/'
        os.makedirs(melspec_path, exist_ok=True)
        a = os.listdir(melspec_path)
        #while(len(a)<100):
        for i, audio in enumerate(self.audio_files):   
            mel_spectrogram = librosa.feature.melspectrogram(y=audio,sr=self.sr,n_mels=130)#TODO n_fft and hop_length to be added
            #print(type(mel_spectrogram))
            #print(melspec_path + ((self.name).split(".")[0]) +"_"+str(i)+'.jpeg' )
            I8 = (mel_spectrogram * 255.9).astype(np.uint8)
            img = Image.fromarray(I8)
            print(np.shape(img))
            img.save(melspec_path + ((self.name).split(".")[0]) +"_"+str(i)+'.png')
            
            #mean = np.mean(mel_spectrogram)
            #range = np.ptp(mel_spectrogram)
            #sd = np.std(mel_spectrogram)
            #min = np.amin(mel_spectrogram)
            #max = np.amax(mel_spectrogram)
            #print(mean,range,sd,min,max)
            #plt.figure(figsize=(3,1))
            #plt.colorbar(format = "%+2.f")
            #librosa.display.specshow(mel_spectrogram,x_axis="time",y_axis="mel",sr=self.sr)
            #exit()
       
                
                           
    def mfccc_extractor(self):
        for  audio in self.audio_files:
             # x, sr = librosa.load(self.chunked_audio_dir + audio)
             mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=39, hop_length=2004)
             scaler = MinMaxScaler(feature_range=(0, 1))
             scaler = scaler.fit(mfccs)
             scaled_mfccs = scaler.transform(mfccs)
             b = scaled_mfccs.tolist()  # ---list of shape 39,12
             c = []
             for i in range(39):
                 row_mean = np.mean(b[i])
                 c.insert(i, row_mean)
             # open the file in the append mode
             with open('res/mfcc_features.csv', 'a') as f:
                 # create the csv writer
                 writer_object = csv.writer(f)
                 # write a row to the csv file
                 writer_object.writerow(c)
                 f.close()

    def melspec_extractor(self):
            #FOR 1 SEC AUDIO(130,44) ---- (number of Mel bins,the number of features)
            melspec_path = 'res/log_melspecs/'
            os.makedirs(melspec_path, exist_ok=True)
            a = os.listdir(melspec_path)
            #while(len(a)<100):
            for i, audio in enumerate(self.audio_files):   
                mel_spectrogram = librosa.feature.melspectrogram(y=audio,sr=self.sr,n_mels=130)#TODO n_fft and hop_length to be added
                #print(type(mel_spectrogram))
                #print(melspec_path + ((self.name).split(".")[0]) +"_"+str(i)+'.jpeg' )
                I8 = (mel_spectrogram * 255.9).astype(np.uint8)
                img = Image.fromarray(I8)
                print(np.shape(img))
                img.save(melspec_path + ((self.name).split(".")[0]) +"_"+str(i)+'.png')
                
                #mean = np.mean(mel_spectrogram)
                #range = np.ptp(mel_spectrogram)
                #sd = np.std(mel_spectrogram)
                #min = np.amin(mel_spectrogram)
                #max = np.amax(mel_spectrogram)
                #print(mean,range,sd,min,max)
                #plt.figure(figsize=(3,1))
                #plt.colorbar(format = "%+2.f")
                #librosa.display.specshow(mel_spectrogram,x_axis="time",y_axis="mel",sr=self.sr)
                #exit()
       
a2r = Audio2Repr()
print(a2r)
