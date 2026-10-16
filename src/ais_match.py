import os
import requests
import pandas as pd
import numpy as np


MAX_DISTANCE_KM = 2.0
TIME_WINDOW_MINUTES = 15
GFW_EVENT_RADIUS_KM = 10.0
GFW_API_TIMEOUT = 20
GFW_EVENT_LIMIT = 100

GFW_EVENTS_URL = (
    "https://gateway.api.globalfishingwatch.org/v3/events"
)
GFW_GAP_DATASET = "public-global-gaps-events:latest"


def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in kilometres."""
    R = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2) - np.radians(lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def normalize_mmsi(value):
    """Normalize MMSI/SSVID values for comparison."""
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value or None


def get_gfw_gap_events(ship_lat, ship_lon, sar_timestamp):
    """
    Retrieve GFW AIS-gap events around the SAR date.

    Events are evidence for investigation, not automatic proof
    that the satellite-detected ship experienced an AIS blackout.
    """

    token = os.getenv("GFW_API_TOKEN")

    if not token:
        return {
            "available": False,
            "events": [],
            "message": "GFW_API_TOKEN is not configured."
        }

    sar_time = pd.to_datetime(
        sar_timestamp, utc=True, errors="coerce"
    )

    if pd.isna(sar_time):
        return {
            "available": False,
            "events": [],
            "message": "Invalid SAR timestamp."
        }

    start_date = (
        sar_time - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    end_date = (
        sar_time + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    payload = {
        "datasets": [GFW_GAP_DATASET],
        "startDate": start_date,
        "endDate": end_date,
        "types": ["GAP"]
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            GFW_EVENTS_URL,
            params={
                "offset": 0,
                "limit": GFW_EVENT_LIMIT
            },
            headers=headers,
            json=payload,
            timeout=GFW_API_TIMEOUT
        )

        if response.status_code not in (200, 201):
            return {
                "available": False,
                "events": [],
                "message": (
                    f"GFW API returned HTTP "
                    f"{response.status_code}."
                )
            }

        data = response.json()
        nearby_events = []

        for event in data.get("entries", []):
            event_start = pd.to_datetime(
                event.get("start"),
                utc=True,
                errors="coerce"
            )

            event_end = pd.to_datetime(
                event.get("end"),
                utc=True,
                errors="coerce"
            )

            # Event must overlap the SAR acquisition time.
            if (
                pd.isna(event_start)
                or pd.isna(event_end)
                or not (event_start <= sar_time <= event_end)
            ):
                continue

            gap = event.get("gap") or {}
            positions = [
                gap.get("onPosition"),
                gap.get("offPosition")
            ]

            distances = []

            for position in positions:
                if not position:
                    continue

                lat = position.get("lat")
                lon = position.get("lon")

                if lat is None or lon is None:
                    continue

                try:
                    distance = float(
                        haversine(
                            ship_lat,
                            ship_lon,
                            float(lat),
                            float(lon)
                        )
                    )
                    distances.append(distance)

                except (TypeError, ValueError):
                    continue

            if not distances:
                continue

            nearest_distance = min(distances)

            if nearest_distance > GFW_EVENT_RADIUS_KM:
                continue

            vessel = event.get("vessel") or {}

            nearby_events.append({
                "event_type": event.get("type"),
                "start": str(event_start),
                "end": str(event_end),
                "vessel_id": vessel.get("id"),
                "vessel_name": vessel.get("name"),
                "ssvid": normalize_mmsi(vessel.get("ssvid")),
                "intentional_disabling": gap.get(
                    "intentionalDisabling"
                ),
                "distance_km": round(nearest_distance, 2)
            })

        return {
            "available": True,
            "events": nearby_events,
            "message": (
                f"GFW API request succeeded. "
                f"Checked {len(data.get('entries', []))} returned "
                f"events; found {len(nearby_events)} nearby "
                f"events overlapping the SAR time. "
                f"API-reported total: {data.get('total', 'unknown')}. "
                f"Results may be incomplete because the query is "
                f"limited to {GFW_EVENT_LIMIT} events."
            )
        }

    except (requests.RequestException, ValueError) as exc:
        return {
            "available": False,
            "events": [],
            "message": (
                f"GFW query failed: {type(exc).__name__}."
            )
        }


def match_ship(
    ship_lat,
    ship_lon,
    sar_timestamp,
    csv_path
):
    """Match a SAR detection against local AIS and GFW evidence."""

    sar_time = pd.to_datetime(
        sar_timestamp, utc=True, errors="coerce"
    )

    if pd.isna(sar_time):
        raise ValueError("Invalid SAR acquisition timestamp.")

    df = pd.read_csv(csv_path)

    required_columns = {
        "timestamp",
        "lat",
        "lon",
        "mmsi",
        "matched_category",
        "matching_score"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "AIS CSV missing columns: "
            + ", ".join(sorted(missing))
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df = df.dropna(
        subset=["timestamp", "lat", "lon"]
    ).copy()

    df["time_difference_min"] = (
        (df["timestamp"] - sar_time)
        .abs().dt.total_seconds() / 60.0
    )

    temporal_df = df[
        df["time_difference_min"] <= TIME_WINDOW_MINUTES
    ].copy()

    gfw = get_gfw_gap_events(
        ship_lat, ship_lon, sar_time
    )

    nearby_events = gfw["events"]

    # No local candidate: do not claim a confirmed blackout
    # solely because a nearby GFW event exists.
    if temporal_df.empty:
        return {
            "status": "Potential Dark Ship",
            "distance_km": None,
            "time_difference_min": None,
            "mmsi": None,
            "category": "N/A",
            "matching_score": None,
            "gfw_api_available": gfw["available"],
            "gfw_gap_event_found": bool(nearby_events),
            "gfw_events": nearby_events,
            "gfw_message": gfw["message"],
            "classification_reason": (
                "No local AIS candidate was found within the "
                "configured time window. A GFW gap event has not "
                "been established as belonging to this detection."
            )
        }

    temporal_df["distance_km"] = haversine(
        ship_lat,
        ship_lon,
        temporal_df["lat"].values,
        temporal_df["lon"].values
    )

    temporal_df["score_distance"] = (
        temporal_df["distance_km"]
        + temporal_df["time_difference_min"] * 0.05
    )

    nearest = temporal_df.loc[
        temporal_df["score_distance"].idxmin()
    ]

    distance = float(nearest["distance_km"])
    time_diff = float(nearest["time_difference_min"])

    mmsi = normalize_mmsi(nearest["mmsi"])

    raw_category = nearest["matched_category"]
    category = (
        "N/A" if pd.isna(raw_category)
        else str(raw_category)
    )

    raw_score = nearest["matching_score"]
    matching_score = (
        None if pd.isna(raw_score)
        else float(raw_score)
    )

    # Require vessel identity to match before using a GFW gap
    # event to support an AIS Blackout classification.
    matched_gap = None

    if mmsi is not None:
        for event in nearby_events:
            if event.get("ssvid") == mmsi:
                matched_gap = event
                break

    if matched_gap is not None:
        status = "AIS Blackout"
        reason = (
            "A GFW gap event overlaps the SAR acquisition time, "
            "is geographically nearby, and its reported SSVID "
            "matches the local AIS candidate."
        )

    elif distance <= MAX_DISTANCE_KM:
        status = "Normal Ship"
        reason = (
            "A local AIS candidate is within the configured "
            "spatial and temporal thresholds. No matching GFW "
            "gap event was identified in the returned records."
        )

    else:
        status = "Potential Dark Ship"
        reason = (
            "The nearest local AIS candidate is outside the "
            "configured spatial threshold, and no vessel-associated "
            "GFW gap event was established."
        )

    return {
        "status": status,
        "distance_km": round(distance, 2),
        "time_difference_min": round(time_diff, 2),
        "mmsi": mmsi,
        "category": category,
        "matching_score": (
            None if matching_score is None
            else round(matching_score, 4)
        ),
        "gfw_api_available": gfw["available"],
        "gfw_gap_event_found": bool(nearby_events),
        "gfw_events": nearby_events,
        "gfw_matched_event": matched_gap,
        "gfw_message": gfw["message"],
        "classification_reason": reason
    }