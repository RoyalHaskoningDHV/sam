import datetime
import math


def _try_parsing_date(text):
    """
    Helper function to try parsing text that either does or does not have a time
    To make the functions below easier, since often time is optional in the apis
    """
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError("No valid date format found")


def _haversine(stations_row, lat2, lon2):
    """
    Helper function to calculate the distance between a station and a (lat, lon) position
    stations_row is a row of knmi_stations, which means it's a dataframe with shape (1, 3)
    `Credit for this solution goes to stackoverflow <https://stackoverflow.com/a/19412565>`
    """
    lat1, lon1 = math.radians(stations_row["latitude"]), math.radians(stations_row["longitude"])
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    a = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return 6373.0 * c  # radius of the earth, in km
