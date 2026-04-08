import numpy as np
import os

class ValidationStats:

  def __init__(self):
    self.KfoldDict = {}
    pass

  def insert_kfold(self, kfoldstats: KfoldStats, fold_id: int):
    self.KfoldDict[fold_id] = kfoldstats
    pass

  def get_kfold(self, fold_id: int) -> KfoldStats:
    assert fold_id in self.KfoldDict, f"Fold ID {fold_id} not found in validation stats dictionary"
    return self.KfoldDict[fold_id]
      

  def save_stats(self, filepath:str):

    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname):
      os.makedirs(dirname)

    np.savez(filepath, **self.KfoldDict)

    self.log(f'Saved metrics to {filepath}')


class KfoldStats:
  def __init__(self, pred: np.array, target: np.array):

    #======MODEL OUTPUT STATS========
    self.plot_preds = []
    self.plot_actual = []
    self.gen_predictions_all = []
    self.rain_predictions_all = []
    self.gen_truth_all = []
    self.rain_truth_all = []

    #======TRAINING STATS==========

  def get_average_rmse(self):
    pass

  def get_average_mse(self):
    pass

  def get_average_mae(self):
    pass

  def get_plot_preds_actual(self):
    pass