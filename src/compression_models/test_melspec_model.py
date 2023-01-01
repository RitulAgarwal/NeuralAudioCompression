import matplotlib.pyplot as plt
import wave
from scipy.io.wavfile import write
from PIL import Image
import os
import cv2
import numpy as np
import sounddevice as sd
import librosa
import keras as keras
import librosa.display
from scipy.io import wavfile
import logging
from numpy.linalg import norm

from moviepy.editor import concatenate_audioclips, AudioFileClip

from train_melspec_compression_model import train_melspec_compression_model 
logging.basicConfig(
    format='[%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

checkpoint_path= "/home/avpl/Documents/audio_related/OOPS_proj/NeuralAudioCompression/checkpoints_mel/39-23.162159.hdf5"
def process_audio(path=None, chunk_time=0.3, loaded_file=None):
    os.makedirs("testAudios", exist_ok=True)
    if path is None:
        return
    x, sr = librosa.load(path)
    n = int(sr * chunk_time)
    count=0
    testAudios = []
    for i in range(0, len(x), n)[:-1] :
        count+=1
        chunk = np.array(x[i:i + n])
        write(("testAudios/" +  str(count) + os.path.basename(path)) , rate=sr, data = chunk)   
        a = str(count) + os.path.basename(path)
        testAudios.append(a)
    return int(count),testAudios

def audio_to_mel(file_path):
    count,testAudios = process_audio(loaded_file=None,chunk_time=0.3,path=file_path)
    testImages=[]
    os.makedirs("testImages", exist_ok=True)
    for i in range(count):
        scale,sr = librosa.load(os.path.join('/home/avpl/Documents/audio_related/OOPS_proj/NeuralAudioCompression/src/compression_models/testAudios',(str(i+1)+'recorded_voice.wav')))
        mel_spectrogram = librosa.feature.melspectrogram(y=scale,sr=sr,n_mels=130)#TODO n_fft and hop_length to be added
        img = (mel_spectrogram * 255).astype(np.uint8)
        new_image = Image.fromarray(img)
        new_image.save("testImages/" + str(i)+'new.png') 
        testImages.append(str(i)+'new.png')
    return count,testImages

model = keras.models.load_model(checkpoint_path)
print(model.summary())

if __name__ == "__main__":
    c,tI = audio_to_mel("/home/avpl/Documents/audio_related/OOPS_proj/NeuralAudioCompression/res/recorded_voice.wav")
    os.makedirs("test_reconst_audios", exist_ok=True)
    final_array = np.empty((130,13))
    for i in tI:
        test_img_path = "/home/avpl/Documents/audio_related/OOPS_proj/NeuralAudioCompression/src/compression_models/testImages/" + str(i)
        ii = cv2.imread(test_img_path)
        gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
        input_arr = keras.utils.img_to_array(gray_image)
        input_arr = np.array([input_arr]) 
        predictions = np.squeeze(model.predict(input_arr)) #130,13

       ## #TODO: 1.concatenate the reconstructed array values 130,13 and so on and then recnstruct the whole audio at once
       # a = np.concatenate((final_array,predictions),axis = 1)
       # final_array=a       
       # audio_signal = librosa.feature.inverse.mel_to_audio(final_array, sr=22050) 
       # wavfile.write('test.wav',data = audio_signal,rate=22050)
       # scale_test_1,sr_test = librosa.load('test.wav')


        #TODO: 2.first reconnstruct all the audios frim their reconstructed audios 
        audio_signal = librosa.feature.inverse.mel_to_audio(predictions, sr=22050) 
        i = i.replace('.png','')
        wavfile.write("test_reconst_audios/" + str(i)+'test.wav' ,data = audio_signal,rate = 22050)
    audios = []
    for i in range(c):
        path = "/home/avpl/Documents/audio_related/OOPS_proj/NeuralAudioCompression/src/compression_models/test_reconst_audios/"
        audio_path = path + str(i) + 'newtest.wav'
        audios.append(audio_path)

    def concatenate_audio_moviepy(audio_clip_paths, output_path):
        clips = [AudioFileClip(c) for c in audio_clip_paths]
        final_clip = concatenate_audioclips(clips)
        final_clip.write_audiofile(output_path)

    concatenate_audio_moviepy(audios,'Test.wav')
    scale_test_2,sr_test = librosa.load('Test.wav')


    #first_shape=scale_test_1.shape[0]
    #second_shape=scale_test_2.shape[0]
    #n = first_shape-second_shape
    #A=scale_test_1[n:]
    #B=scale_test_2
    #cosine = np.dot(A,B)/(norm(A)*norm(B))
    #print("Cosine Similarity:", cosine)
    #    





        


    ###batch_Ytest = []
    ###
    ####image = keras.utils.load_img(test_img_path)
    ####input_arr = keras.utils.img_to_array(image)
    ####logging.info(input_arr.shape)
    #### img_gray = image.convert('L')
    #### img_gray.save('/home/avpl/Documents/audio_related/OOPS_proj/NeuralAudioCompression/src/compression_models/testImages/2NEW_new.png')
    #### image = keras.utils.load_img('/home/avpl/Documents/audio_related/OOPS_proj/NeuralAudioCompression/src/compression_models/testImages/2NEW_new.png')
    #### input_arr = keras.utils.img_to_array(image)
    #### logging.info(input_arr.shape)
    ###ii = cv2.imread(test_img_path)
    ###gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
    ###input_arr = keras.utils.img_to_array(gray_image)
    ###input_arr = np.array([input_arr]) 
    ###logging.info(input_arr.shape)
    ####plt.imshow(gray_image,cmap='Greys')
    ####plt.show()
###
    ###predictions = np.squeeze(model.predict(input_arr))
    ###print(predictions,type(predictions),predictions.shape)
###
    ####new_image = Image.fromarray(predictions)
    ####if new_image.mode != 'RGB':
    ####    new_image = new_image.convert('RGB')
    ####new_image.save('new.png') 
    ###audio_signal = librosa.feature.inverse.mel_to_audio(predictions, sr=22050) 
    ###wavfile.write('test.wav',data = audio_signal,rate=22050)
    ###scale_test,sr_test = librosa.load('test.wav')
    ###print(scale_test, scale_test.shape)
    ###  
    #### saving the final output 
    #### as a PNG file
    ####batch_Ytest.append(img)
###
    #### for i in range(c):
    ####     preds = model.predict('/home/avpl/Documents/audio_related/OOPS_proj/NeuralAudioCompression/src/compression_models/testAudios/'+tA[i+1])
    ####     print(type(preds),preds.shape)
###



