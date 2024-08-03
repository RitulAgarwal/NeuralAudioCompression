
#  Audio Conversion through Embedded software

A Deep Learning project using convolutional autoencoder network, that takes in input as an audio (possibly through a mobile app)
and produces the corresponding vibrational values for a hand band for deaf people that's embedded with actuators. 

 


## Visualizing The Pipeline

![App Screenshot](https://i.postimg.cc/wvJqyTxj/Screenshot-2024-08-03-172900.png)


## Installation


We have provided the model checkpoint that can be used as follows.

```bash
  cd checkpoints_mel
```
Now the model can be loaded as follows.

```bash
  from keras.models import load_model
  model = load_model('model.h5')
```
Our model Pipeline : 
```bash
  cd src/pipeline.py
```

The pipeline encapsulates our model. 
You can train your own model on your audio data through this file.

```bash
  cd src/compression_models/train_melspec_compression_model.py
```

Additionally, hyperparamter tuning through RayTune has been done and can be tweaked as per the user requirements through 

```bash
  python main_mel_ray_1.py
  python main_mel_raytune.py

```


## Tech Stack

Python, Keras, PlatformIO, Librosa, Raytune


## Related

Some related projects

[Speech2Speech Convertor](https://github.com/RitulAgarwal/DiffAudTrRepo)




## Contributing


Contributions are always welcome!
To add any additional functionality, or to indicate out errors in the code you may raise an issue. 
## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ritulagarwal.github.io/portfolio/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ritul-agarwal1702/)


