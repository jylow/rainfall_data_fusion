import optuna
from src.utils import read_config
import copy
from train_fused import train_fused

def objective(trial):
    config = read_config("config.yaml")

    config["layer_connect"]["gauge_gauge"] = trial.suggest_int(
        "gauge_gauge", 1, 10
    )
    config["layer_connect"]["cml_gauge"] = trial.suggest_int(
        "cml_gauge", 1, 15
    )
    config["model"]["hidden_channels"] = trial.suggest_categorical(
        "hidden_channels", [8, 16, 32, 64, 128]
    )
    config["model"]["num_layers"] = trial.suggest_int(
        "num_layers", 1, 16
    )
    config["model"]["learning_rate"] = trial.suggest_float(
        "learning_rate", 0.0001, 0.01, log=True
    )
    config["model"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 0.000001, 0.1, log=True
    )
    config["dataset_parameters"]["weighted_loss_alpha"] = trial.suggest_float(
        "weighted_loss_alpha", 0, 1, log=False
    )
    config["dataset_parameters"]["include_lpe"] = trial.suggest_categorical(
        "include_lpe", [True, False]
    )

    # Returns multiple metrics
    metrics_dict = train_fused(config)

    for key, val in metrics_dict.items():
        if key != "per_station_metrics":          # skip non-scalar
            trial.set_user_attr(key, val)

    # Return only the 3 optimisation targets as a tuple
    return (
        metrics_dict["rmse"],            # minimize
        #metrics_dict["timestep_rmse"],   # minimize
        #metrics_dict["f1"],              # maximize
    )


study = optuna.create_study(
    directions=["minimize"],
    #directions=["minimize", "minimize", "maximize"],
    study_name="rain_gnn_multiobjective",
    storage="sqlite:///optuna_multi_raingauge_cml.db",
    load_if_exists=True,
    sampler=optuna.samplers.NSGAIISampler(seed=42),
)

study.optimize(objective, n_trials=50, n_jobs=1)
