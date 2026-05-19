import pandas as pd
from datetime import datetime
from torch_geometric.data import HeteroData


def load_raingauge_dataset(
    start:int,
    end: int,
    uptime_threshold: float = 1.0,
    folder_path: str = 'database/raingauge_nea_data'
) -> tuple:
    """
    Loads raingauge dataset into a pandas DataFrame object
    ------
    dataset_name: .csv file
    """

    #Process raingauge df
    print(f"Loading raingauge data from {start} to {end}")
    complete_raingauge_df = []
    for year in range(start, end + 1):
        filepath = f"{folder_path}/{year}/weather_station_data_{year}.csv"
        print(f"Loading raingauge_dataset from {filepath}")
        raingauge_df = pd.read_csv(filepath)
        complete_raingauge_df.append(raingauge_df)

    gauge_df = pd.concat(complete_raingauge_df)


    gauge_df["timestamp"] = gauge_df["timestamp"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:00+08:00")
    )
    gauge_df['value'] = gauge_df['value'] * 12
    formatted_gauge_df = gauge_df.pivot(
        index="timestamp", columns="stationId", values="value"
    )
    print("Loading complete")
    print(f"Dataframe shape: {formatted_gauge_df.shape}")

    #Filter for threshold
    if uptime_threshold:
        print(f"Filtering raingauge_uptime. Threshold = {uptime_threshold}")
        filtered_stations_index = filter_uptime(formatted_gauge_df, uptime_threshold=uptime_threshold).index
        formatted_gauge_df = formatted_gauge_df[filtered_stations_index]

    #Get raingauge coordinate mappings
    print("Getting raingauge coordinate mappings")
    raingauge_mappings_df = get_station_mapping_df(start=start, end=end)
    raingauge_mappings_df = raingauge_mappings_df[raingauge_mappings_df['id'].isin(filtered_stations_index)]
    print(f"Mapping df shape: {raingauge_mappings_df.shape}")
    raingauge_mappings_df['order'] = [i for i in range(raingauge_mappings_df.shape[0])]
    raingauge_mappings_df.reset_index(inplace=True)

    return formatted_gauge_df, raingauge_mappings_df



def filter_uptime(raingauge_df: pd.DataFrame, uptime_threshold = 0.9) -> pd.DataFrame:
    '''
    Filters dataframe for threshold where we keep only stations with >threshold uptime

    :param df: Description
    :return: Description
    :rtype: DataFrame
    '''
    raingauge_uptime = raingauge_df.notna().sum() / len(raingauge_df)
    filtered_stations_df = raingauge_uptime[raingauge_uptime >= uptime_threshold]
    return filtered_stations_df



def get_station_coordinate_mappings(filename="database/weather_stations.csv", start: int = 0, end: int = 0) -> dict:
    """
    Returns dictionary containing the mappings of station names to coordinates for raingauge

    dict: [key, (lat,lon)]
    ------
    """

    station_df = pd.DataFrame()

    for year in range(start, end + 1):
        df = pd.read_csv(f"database/raingauge_nea_data/{year}/weather_stations_{year}.csv")
        station_df = pd.concat([station_df, df]).drop_duplicates(['id', 'latitude', 'longitude']).reset_index(drop=True)
    station_dict = dict(zip(station_df['id'], zip(station_df['latitude'], station_df['longitude'])))
    return station_dict



def get_station_mapping_df(start: int, end: int) -> pd.DataFrame:
    station_df = pd.DataFrame()

    for year in range(start, end + 1):
        df = pd.read_csv(f"database/raingauge_nea_data/{year}/weather_stations_{year}.csv")
        station_df = pd.concat([station_df, df]).drop_duplicates(['id', 'latitude', 'longitude']).reset_index(drop=True)

    station_df.drop(columns='deviceId', inplace=True)
    return station_df
