import pandas as pd
import numpy as np


MAX_DISTANCE_KM = 2.0
TIME_WINDOW_MINUTES = 15


def haversine(lat1, lon1, lat2, lon2):

    R = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


def match_ship(
    ship_lat,
    ship_lon,
    sar_timestamp,
    csv_path
):

    df = pd.read_csv(csv_path)

    # Convert timestamps
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    sar_time = pd.to_datetime(
        sar_timestamp,
        utc=True
    )

    # Remove invalid timestamps
    df = df.dropna(
        subset=["timestamp"]
    )

    # ---------------------------------------------------------
    # TEMPORAL FILTER
    # ---------------------------------------------------------

    time_difference = (
        (df["timestamp"] - sar_time)
        .abs()
        .dt.total_seconds()
        / 60.0
    )

    temporal_df = df[
        time_difference <= TIME_WINDOW_MINUTES
    ].copy()

    # No AIS/GFW information in the required time window
    if temporal_df.empty:

        return {
            "status": "Dark Ship",
            "distance_km": None,
            "time_difference_min": None,
            "mmsi": None,
            "category": "N/A",
            "matching_score": None
        }

    # ---------------------------------------------------------
    # SPATIAL DISTANCE
    # ---------------------------------------------------------

    temporal_df["distance_km"] = haversine(
        ship_lat,
        ship_lon,
        temporal_df["lat"].values,
        temporal_df["lon"].values
    )

    # ---------------------------------------------------------
    # FIND BEST SPATIAL + TEMPORAL CANDIDATE
    # ---------------------------------------------------------

    temporal_df["score_distance"] = (
        temporal_df["distance_km"]
        + temporal_df["timestamp"]
        .sub(sar_time)
        .abs()
        .dt.total_seconds()
        / 60.0
        * 0.05
    )

    nearest = temporal_df.loc[
        temporal_df["score_distance"].idxmin()
    ]

    distance = float(
        nearest["distance_km"]
    )

    time_diff = abs(
        (
            nearest["timestamp"]
            - sar_time
        ).total_seconds()
    ) / 60.0

    # ---------------------------------------------------------
    # FINAL CLASSIFICATION
    # ---------------------------------------------------------

    if distance <= MAX_DISTANCE_KM:

        status = "Normal Ship"

    else:

        status = "Dark Ship"

    # ---------------------------------------------------------
    # INFORMATION
    # ---------------------------------------------------------

    if pd.isna(nearest["mmsi"]):

        mmsi = None

    else:

        mmsi = int(
            nearest["mmsi"]
        )

    if pd.isna(nearest["matched_category"]):

        category = "N/A"

    else:

        category = str(
            nearest["matched_category"]
        )

    if pd.isna(nearest["matching_score"]):

        matching_score = None

    else:

        matching_score = float(
            nearest["matching_score"]
        )

    return {

        "status": status,

        "distance_km": round(
            distance,
            2
        ),

        "time_difference_min": round(
            time_diff,
            2
        ),

        "mmsi": mmsi,

        "category": category,

        "matching_score": (
            None
            if matching_score is None
            else round(
                matching_score,
                4
            )
        )
    }