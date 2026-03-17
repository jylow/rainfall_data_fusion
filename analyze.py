# analyse_results.py — run this locally or on a login node
import optuna
import pandas as pd
from optuna.storages import RDBStorage

storage = RDBStorage(
    url="sqlite:////home/l/lowjunyu/rainfall_data_fusion/optuna_multi_all.db",
    engine_kwargs={"connect_args": {"timeout": 30}}
)

study = optuna.load_study(
    study_name="rain_gnn_multiobjective",
    storage=storage
)

# --- Pareto front ---
print(f"\n{len(study.best_trials)} Pareto-optimal trials:\n")
for t in study.best_trials:
    print(f"  Trial {t.number:>3} | "
          f"RMSE={t.values[0]:.4f}  "
          f"ts_RMSE={t.values[1]:.4f}  "
          f"F1={t.values[2]:.4f}")
    print(f"             | {t.params}\n")

# --- Full results to CSV ---
df = study.trials_dataframe().rename(columns={
    "values_0": "rmse",
    "values_1": "timestep_rmse",
    "values_2": "f1",
})
df.to_csv("results/all_trials.csv", index=False)

# --- Live dashboard (run on login node, open in browser) ---
# optuna-dashboard sqlite:////your/nfs/path/results/optuna.db
