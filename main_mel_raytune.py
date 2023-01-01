# import os
# import librosa
# from pydub import AudioSegment
# from scipy.io.wavfile import write
# from src.audio2repr import Audio2Repr
# import logging
# import pandas as pd
# import numpy as np
# from ray import tune
# from ray.tune.search.hyperopt import HyperOptSearch
# logging.basicConfig(
#     format='[%(filename)s:%(lineno)d] %(message)s',
#     datefmt='%Y-%m-%d:%H:%M:%S',
#     level=logging.INFO)
# import csv
from src.compression_models.train_melspec_compression_model import train_melspec_compression_model  
from ray.air import session


# def objective(args,config):
#     lr,batch_size,tmcm = args
#     hs = tmcm.train_model()
#     return ({"mae": max(hs.history['val_loss'])})

# def call_melspec_pipeline():
#     tmcm = train_melspec_compression_model('/home/avpl/Documents/audio_related/OOPS_proj/OneSecAudios',
#                                             config={
#                                                         'batch_size':'16',
#                                                         'lr':'0.09'
#                                                     }
#                                         )
#     return tmcm
    


# if __name__ == "__main__":  
#     sr = 22050
#     mel_model = call_melspec_pipeline()

#     # Setup state space
#     search_space = {
# 	        "lr": tune.uniform(0.0001,0.1),
# 	        "batch_size": tune.choice( [8, 16, 32, 64]),
#             "loaded_data" : tune.choice([mel_model])
#         }
    
#     # Declare objective function above - call train_model function and return best metric
    

#     # Call ray tune optimiser to optimise on the objective function
#     tuner = tune.Tuner(
#     objective,
#     tune_config=tune.TuneConfig(
#         metric="accuracy",
#         mode="max",
#         search_alg=algo,
#     ),
#     param_space=search_space,
# )
# results = tuner.fit()
#     def ray_tune_optimi(self):
        
#         algo = HyperOptSearch()

#         tuner = tune.Tuner(
#             objective,
#             search_algo = algo,
#             param_space=search_space,
#             ),
        
#         results = tuner.fit()  
#         return (results.get_best_result(metric="mae",mode="min").config)

#
#
#from ray import tune
#from ray.tune.search.hyperopt import HyperOptSearch
#import keras
#
#
## 1. Wrap a Keras model in an objective function.
#def objective(config):
#    # model = config["model"]
#    # lr = config["lr"]
#    # batch_size = config["batch_size"]
#    
#    hist = config["loaded_data"].train_model(bs = config["batch_size"], lr = config["lr"])
#    #return {"accuracy": hist}
#    return {"mae": min(hist.history['val_loss'])}
#
#def get_model():
#    tmcm = train_melspec_compression_model('/home/avpl/Documents/audio_related/OneSecAudios',
#                                            config={
#                                                        'batch_size':'16',
#                                                        'lr':'0.09'
#                                                    }
#                                        )
#    return tmcm
#
## 2. Define a search space and initialize the search algorithm.
#algo = HyperOptSearch()
#
#search_space = {
#        "lr": tune.uniform(0.000001,0.001),
#        "batch_size": tune.choice( [8, 16, 32, 64]),
#        "loaded_data" : tune.choice([get_model()])
#    }
#
#0
## 3. Start a Tune run that maximizes accuracy.
#tuner = tune.Tuner(
#    objective,
#    tune_config=tune.TuneConfig(
#        metric="mae",
#        mode="min",
#        search_alg=algo,
#        num_samples=10
#    ),
#    param_space=search_space, 
#)
#result_grid = tuner.fit()
#
## Get last reported results per trial
#df = result_grid.get_dataframe()
#
## Get best ever reported accuracy per trial
#df = result_grid.get_dataframe(
#    filter_metric="mae", filter_mode="min"
#)
#
##best_result = result_grid.get_best_result( 
##    metric="loss", mode="min")
##print("BEST RESULT IS !!!!",best_result)
#print("OUR DATAFRAME IS !!@@@???E",df)
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler as ASHAScheduler
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray import air
def get_model():
    tmcm = train_melspec_compression_model('/home/avpl/Documents/audio_related/OneSecAudios', 
                                            config = {
                                                    "lr": tune.uniform(0.000001,0.001),
                                                    "batch_size": tune.choice( [8, 16, 32, 64]),
                                                    #"loaded_data" : tune.choice([get_model()])
                                                }
                                        )
    return tmcm
config = {
        "lr": tune.uniform(0.000001,0.001),
        "batch_size": tune.choice( [8, 16, 32, 64]),
        #"loaded_data" : tune.choice([get_model()])
    }
# algo = HyperOptSearch()
# def objective(bs,lr):
#     data_model = get_model()
#     hist = data_model.train_model(bs, lr)
#     #return {"accuracy": hist}
#     return (hist.history['val_loss'])

asha_scheduler = ASHAScheduler(
    time_attr='num_training_iterations',
    # metric='score',
    # mode='min',
    max_t=10,
    grace_period=8,
    #reduction_factor=3,
    #brackets=1)
)

def trainable(config):
    data_model = get_model()
    final_loss=data_model.train_model(config["batch_size"],config["lr"]).history['val_loss']
    session.report({"score": final_loss}) 
    return {"score": final_loss}
    
# tuner = tune.Tuner(trainable, 
#             tune_config=tune.TuneConfig(
#             metric="score",
#             mode="min",
#             #scheduler=asha_scheduler,
#             search_alg=algo,
#             num_samples=10), 
#             run_config=air.RunConfig(stop=
#                                         {"training_iteration": 7,
#                                         "score": 30}),
#             param_space=config,
# )
# results = tuner.fit()

# #df_results = results.get_dataframe()
# #df = results.get_dataframe(filter_metric="score", filter_mode="max")

# #best_result = results.get_best_result()  # Get best result object
# #best_config = best_result.config  # Get best trial's hyperparameters
# print("Best hyperparameters found were: ", results.get_best_result().config)

from ray.tune.schedulers import AsyncHyperBandScheduler

def tune_mnist(num_training_iterations):
    
    
    tuner = tune.Tuner(
        trainable ,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="min",
            scheduler=asha_scheduler,
            #search_alg=algo,
            num_samples=10), 
        run_config=air.RunConfig(stop=
                                        {"training_iteration": num_training_iterations,
                                        "score": 30}),
        
        param_space=config,
    )
    results = tuner.fit()
    return results 

    

a = tune_mnist(num_training_iterations=5)

if (a.get_best_result(metric="score", mode="min")):
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Best hyperparameters found were: ", a.get_best_result(metric="score", mode="min").config)
else:
    print("nothing$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")