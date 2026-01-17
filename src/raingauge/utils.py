import pandas as pd
from datetime import datetime


def load_raingauge_dataset(
    filepath: str
) -> pd.DataFrame:
    """
    Loads raingauge dataset into a pandas DataFrame object
    ------
    dataset_name: .csv file
    """
    print("Loading raingauge_dataset from {filepath}")
    gauge_df = pd.read_csv(filepath)

    print(gauge_df.iloc[0])

    # format time
    gauge_df["timestamp"] = gauge_df["timestamp"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:00+08:00")
    )

    # convert to table with stations as columns
    formatted_gauge_df = gauge_df.pivot(
        index="timestamp", columns="stationId", values="value"
    )
    print("Loading complete")
    print(f"Dataframe shape: {formatted_gauge_df.shape}")
    return formatted_gauge_df

'''
DEPRECIATED

def load_weather_station_dataset(
    dataset_name: str, dataset_folder="database"
) -> pd.DataFrame:
    """
    Loads weather station dataset(CSV) into a pandas DataFrame object
    ------
    dataset_name: .csv file
    """

    path = f"{dataset_folder}/{dataset_name}"
    gauge_df = pd.read_csv(path)

    # format time
    gauge_df.rename(
        columns={"timestamp": "time_sgt", "station_id": "gid"}, inplace=True
    )
    gauge_df["time_sgt"] = gauge_df["time_sgt"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:00+08:00")
    )
    #gauge_df['time_sgt'] = gauge_df['time_sgt'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:00'))

    # convert to table with stations as columns
    filtered_res = gauge_df

    return filtered_res


def get_gauge_coordinate_mappings(filename="database/weather_stations.csv") -> dict:
    """
    Returns dictionary containing the mappings of station names to coordinates for raingauge

    dict: [key, (lat,long)]
    ------
    """

    gauge_df = pd.read_csv(filename)
    station_locations_df = get_gauge_stations()
    station_locations = station_locations_df["gid"].to_numpy()
    station_name_to_coordinates = station_locations_df[
        ["gid", "latitude", "longitude"]
    ].to_numpy()
    station_dict = dict()

    for name, lat, long in station_name_to_coordinates:
        station_dict[name] = (lat, long)

    gauge_df = gauge_df[gauge_df["gid"].isin(station_locations)]

    return station_dict


def get_gauge_stations(filename="database/weather_stations.csv") -> pd.DataFrame:
    station_locations_df = pd.read_csv(filename)

    return station_locations_df


def get_station_coordinate_mappings(filename="database/weather_stations.csv") -> dict:
    """
    Returns dictionary containing the mappings of station names to coordinates for raingauge

    dict: [key, (lat,lon)]
    ------
    """

    gauge_df = pd.read_csv(filename)
    station_locations_df = get_gauge_stations(filename)
    station_locations = station_locations_df["gid"].to_numpy()
    station_name_to_coordinates = station_locations_df[
        ["gid", "latitude", "longitude"]
    ].to_numpy()
    station_dict = dict()

    for name, lat, lon in station_name_to_coordinates:
        station_dict[name] = (lat, lon)

    gauge_df = gauge_df[gauge_df["gid"].isin(station_locations)]

    return station_dict


def get_weather_stations(filename="database/weather_stations.csv") -> pd.DataFrame:
    station_location_df = pd.read_csv(filename)

    return station_location_df
'''