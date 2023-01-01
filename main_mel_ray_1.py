from src.compression_models.train_melspec_compression_model import train_melspec_compression_model
from ray.air import session
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler as ASHAScheduler
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch

from ray import air
config = {
    "lr": tune.uniform(0.000001, 0.001),
    "batch_size": tune.choice([8, 16, 32, 64]),
}


def get_model():
    tmcm = train_melspec_compression_model(
        '/home/avpl/Documents/audio_related/OneSecAudios', config)
    return tmcm
# algo = HyperOptSearch()


def trainable(config):
    data_model = get_model()
    final_loss = min(data_model.train_model(
        config["batch_size"], config["lr"]).history['val_loss'])
    session.report({"score": final_loss})
    # return {"score": final_loss}


def tune_mnist(num_training_iterations):
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=15,
        grace_period=8,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            resources={"cpu": 2, "gpu": 0}
        ),
        tune_config=tune.TuneConfig(
            metric="score",
            mode="min",
            scheduler=asha_scheduler,
            num_samples=8),
        run_config=air.RunConfig(stop={"training_iteration": num_training_iterations,
                                       "score": 30}),

        param_space=config,
    )
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)


if __name__ == "__main__":
    tune_mnist(num_training_iterations=5)

