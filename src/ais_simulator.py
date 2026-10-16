"""
Simulated AIS scenarios for Dark Ship Detection demonstrations.

These are synthetic demonstration results, not real-world AIS findings.
Each detected ship receives a deterministic scenario based on its image
filename and detection number.
"""

from pathlib import Path
import hashlib

import pandas as pd


DEFAULT_METADATA = Path("data/ais/ship_labels_metadata.csv")
DEFAULT_SIMULATED_AIS = Path("data/ais/simulated_ais.csv")

SCENARIOS = (
    "Normal Ship",
    "Dark Ship",
    "AIS Blackout",
    "Potential Dark Ship",
)

SCENARIO_NOTES = {
    "Normal Ship": "Simulated AIS transmission is present.",
    "Dark Ship": "Simulated scenario: no AIS transmission is present.",
    "AIS Blackout": "Simulated scenario: an AIS gap event is assigned.",
    "Potential Dark Ship": "Simulated scenario: AIS verification is unavailable.",
}

# Fixed base scenarios for your five prepared test images.
DEMO_SCENARIOS = {
    "lat02442838lon-7945241__time20221110": "Normal Ship",
    "lat02462130lon-7978721__time20221110": "Dark Ship",
    "lat02480483lon-7970996__time20200804": "AIS Blackout",
    "lat02493796lon-7960086__time20200804": "Potential Dark Ship",
    "lat02510766lon-7967182__time20231209": "Normal Ship",
}


def _find_column(df, candidates):
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def generate_simulated_ais(
    metadata_path=DEFAULT_METADATA,
    output_path=DEFAULT_SIMULATED_AIS,
):
    """Generate a simulated AIS CSV using the available metadata."""

    metadata_path = Path(metadata_path)
    output_path = Path(output_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    df = pd.read_csv(metadata_path)

    file_col = _find_column(
        df, ["file_name", "filename", "image_name"]
    )
    time_col = _find_column(
        df, ["timestamp_s1", "timestamp", "time"]
    )
    mmsi_col = _find_column(
        df, ["vessel_mmsi", "mmsi", "ssvid"]
    )
    name_col = _find_column(
        df, ["vessel_name", "name"]
    )
    lat_col = _find_column(
        df, ["vessel_latitude", "lat", "latitude"]
    )
    lon_col = _find_column(
        df, ["vessel_longitude", "lon", "longitude"]
    )

    if file_col is None:
        raise ValueError(
            "Metadata needs a file_name, filename, or image_name column."
        )

    rows = []

    for index, row in df.iterrows():
        filename = str(row[file_col]).strip()
        stem = Path(filename).stem.lower()

        # Preserve explicitly configured scenarios.
        # Other metadata rows receive a deterministic scenario.
        scenario = DEMO_SCENARIOS.get(
            stem,
            SCENARIOS[index % len(SCENARIOS)],
        )

        rows.append({
            "file_name": filename,
            "timestamp": row[time_col] if time_col else None,
            "mmsi": (
                str(row[mmsi_col]).strip()
                if mmsi_col and pd.notna(row[mmsi_col])
                else ""
            ),
            "vessel_name": (
                str(row[name_col]).strip()
                if name_col and pd.notna(row[name_col])
                else ""
            ),
            "lat": row[lat_col] if lat_col else None,
            "lon": row[lon_col] if lon_col else None,
            "simulated_status": scenario,
            "is_simulated": True,
            "simulation_note": SCENARIO_NOTES[scenario],
        })

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )

    return str(output_path)


def get_detection_scenario(image_name, detection_index):
    """
    Return a deterministic scenario for one detected ship.

    detection_index is zero-based:
        0 = first detected ship
        1 = second detected ship
        ...
    """

    filename = Path(str(image_name)).stem.lower()

    base_scenario = DEMO_SCENARIOS.get(
        filename,
        "Normal Ship",
    )

    # Rotate through scenarios starting from the image's base scenario.
    start_index = SCENARIOS.index(base_scenario)

    scenario_index = (
        start_index + int(detection_index)
    ) % len(SCENARIOS)

    status = SCENARIOS[scenario_index]

    return {
        "status": status,
        "is_simulated": True,
        "simulation_note": SCENARIO_NOTES[status],
        "scenario_id": (
            f"{filename}:detection_{int(detection_index) + 1}"
        ),
    }


def match_simulated_ship(
    image_name,
    simulated_csv=DEFAULT_SIMULATED_AIS,
):
    """
    Return image-level simulated metadata for compatibility
    with the existing dashboard.
    """

    simulated_csv = Path(simulated_csv)

    if not simulated_csv.exists():
        generate_simulated_ais(
            output_path=simulated_csv,
        )

    df = pd.read_csv(simulated_csv)

    file_col = _find_column(df, ["file_name"])

    if file_col is None:
        raise ValueError(
            "Simulated AIS CSV has no file_name column."
        )

    target = Path(str(image_name)).name.lower()
    target_stem = Path(target).stem.lower()

    names = df[file_col].astype(str).map(
        lambda value: Path(value).name.lower()
    )

    matches = df[names == target]

    if matches.empty:
        matches = df[
            names.map(
                lambda value: Path(value).stem.lower()
            ) == target_stem
        ]

    if matches.empty:
        return {
            "status": "Insufficient Evidence",
            "is_simulated": True,
            "simulation_note": (
                "No matching image metadata was found. "
                "Per-detection demo scenarios can still be generated."
            ),
            "mmsi": None,
            "vessel_name": None,
            "timestamp": None,
            "lat": None,
            "lon": None,
        }

    row = matches.iloc[0]

    def value(key):
        item = row.get(key)

        if item is None or pd.isna(item):
            return None

        return item

    return {
        "status": str(row["simulated_status"]),
        "is_simulated": True,
        "simulation_note": str(row["simulation_note"]),
        "mmsi": (
            None if value("mmsi") is None
            else str(value("mmsi"))
        ),
        "vessel_name": (
            None if value("vessel_name") is None
            else str(value("vessel_name"))
        ),
        "timestamp": (
            None if value("timestamp") is None
            else str(value("timestamp"))
        ),
        "lat": (
            None if value("lat") is None
            else float(value("lat"))
        ),
        "lon": (
            None if value("lon") is None
            else float(value("lon"))
        ),
    }