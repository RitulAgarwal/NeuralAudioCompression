import os
import librosa
from pydub import AudioSegment
from scipy.io.wavfile import write
from src.audio2repr import Audio2Repr
import logging
import pandas as pd
import numpy as np
logging.basicConfig(
    format='[%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
import csv
#from src.repr2audio import Repr2Audio
from src.compression_models.melspec1 import train_melspec_compression_model
#from compression_models.train_mfcc_compression_model import train_mfcc_compression_model
#from src.pipeline import Pipeline

#2pipelines

def mp3_to_wav(file_name): 
    input_file = file_name
    output_file = file_name + ".wav"
    sound = AudioSegment.from_mp3(input_file)
    sound.export(output_file, format="wav")
    return output_file


def process_audio(path=None, chunk_time=0.3, loaded_file=None):
    os.makedirs("OneSecAudios", exist_ok=True)
    if path is None:
        return
    x, sr = librosa.load(path)
    print(x,sr)
    n = int(sr * chunk_time)
    print(n)
    for i in range(0, len(x), n)[:-1] :
        chunk = np.array(x[i:i + n])
        write(("OneSecAudios/" +  str(i) + os.path.basename(path)) , rate=sr, data = chunk)    
    return

#def make_list_of_csv():
#    file_name_path = '/home/avpl/Documents/audio_related/OOPS_proj/NeuralAudioCompression/OneSecAudios'
#    df = pd.DataFrame()
#    list_of_files = [os.path.join(file_name_path,i) for i in os.listdir("OneSecAudios")]
#    df["Filenames"] = pd.Series(list_of_files)
#    df.to_csv('res/log_mels.csv', index=None)
#

#def call_mfcc_pipeline(list_of_wavs):
#    for i in list_of_wavs:
#        processed_audio_chunks, sr = process_audio(path = "res/" + i, chunk_time=1)
#        a2r = Audio2Repr(processed_audio_chunks, sr)
#        mfcc_audio2repr = a2r.mfcc_extractor
#
#    tcm = train_mfcc_compression_model('res/mfcc_features.csv',2048)
#    mfcc_trainer = tcm.train_model()
#    r2a = Repr2Audio(processed_audio_chunks, sr)
#    mfcc_pipeline = Pipeline(
#                            audio2repr = None, 
#                            compression_model = tcm, 
#                            repr2audio = r2a)

#NOW WE AMDE CHUNKS SO LIST OF WAVS IS NTO REQD
#def call_melspec_pipeline(list_of_wavs):
    ####FOR MULTIPLE WAV FILES OF MANY SECONDS THEN WE NEED TO PREPROCESS EACH LONG AUDIO FOR ! SEC AND THEN CALCULTAE
    #for i in list_of_wavs:
    #    process_audio(path = i, chunk_time=0.3)
    #    #all_chunks = all_chunks+processed_audio_chunks
    #
    #make_list_of_csv()
def call_melspec_pipeline():
    tmcm = train_melspec_compression_model('/home/avpl/Documents/audio_related/OOPS_proj/OneSecAudios',64
                                        )
    tmcm.train_model()


        #a2r = Audio2Repr(processed_audio_chunks, sr,audio_name)
        #melspec_audio2repr = a2r.melspec_extractor()
        #tmcm = train_melspec_compression_model('res/log_mels.csv',2)
        #log_mel_trainer = tmcm.train_model()

    

    ###DIRECT 1SEC AUDIO IS INPUTTED
    #a2r = Audio2Repr(list_of_wavs,sr)
    #melspec_audio2repr = a2r.melspec_extractor()

    #tcm = train_melspec_compression_model('res/log_melspecs')
    #melspec_trainer = tcm.train_model()
    #r2a = Repr2Audio(processed_audio_chunks, sr)
    #melspec_pipeline = Pipeline(
    #                        audio2repr = None, 
    #                        compression_model = tcm, 
    #                        repr2audio = r2a)


if __name__ == "__main__":
    ###already done into chunks so removed for now 
    ####sfp = "res/confirm_audios/"
    #####when files are mp3 then use this line 
    ####our_wavs = []
    ####for i in os.listdir(sfp) :
    ####    if i.endswith(".mp3"):
    ####        print(os.path.join(sfp, i))
    ####        our_wavs.append(mp3_to_wav(os.path.join(sfp, i)))
    ####print(our_wavs)

    #num_of_audios = len(os.listdir(sfp))
    #########used for the recordde or 1 second audio files that are wav 
    #our_wavs = [os.path.join(sfp, i) for i in os.listdir(sfp)]
    sr = 22050
    #new funtion without wavs
    #call_melspec_pipeline(our_wavs)

    call_melspec_pipeline()
    #mfcc_repr2audio = a2r.mfcc_extractor


# 145x340 for 1 sec
# 