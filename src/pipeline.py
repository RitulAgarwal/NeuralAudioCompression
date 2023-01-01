class Pipeline:
    def __init__(self, audio2repr, repr2audio, compression_model):
        audio2repr()
        compression_model.train_model()
        repr2audio()



