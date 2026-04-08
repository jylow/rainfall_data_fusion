import pandas as pd

from radar.utils import load_radar_dataset

def load_dataset(
    radar_filepath: str,
    raingauge_filepath: str,
    cml_filepath: str
) -> pd.DataFrame:
  '''
  Prepares the dataset for CML, Radar and Rain Gauge
  Returns: Dataframe with all data sources aligned in 15 minute timestep
  '''

  # Prepare radar grid data
  grid_data = load_radar_dataset(radar_filepath)
  return grid_data

prepare_dataset(
    radar_filepath="sg_radar_data",
    cml_filepath="",
    raingauge_filepath=""
  )
