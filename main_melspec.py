import logging
logging.basicConfig(
    format='[%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
from src.compression_models.train_melspec_compression_model import train_melspec_compression_model  

def call_melspec_pipeline():
    tmcm = train_melspec_compression_model('/home/avpl/Documents/audio_related/OneSecAudios',
                                            config={
                                                        'batch_size':'16',
                                                        'lr':'0.0003410346690618139'
                                                    }
                                        )
    tmcm.train_model()


if __name__ == "__main__":
    sr = 22050
    call_melspec_pipeline()
