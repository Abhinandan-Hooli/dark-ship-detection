import os
import xml.etree.ElementTree as ET

import numpy as np
from rasterio.control import GroundControlPoint
from rasterio.transform import GCPTransformer
from ultralytics import YOLO

from ais_match import match_ship


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_balanced_model.pt"
)

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "sentinel1",
    "ship_patch_vv.png"
)

XML_PATH = os.path.join(
    BASE_DIR,
    "data",
    "sentinel1",
    "iw-vv.xml"
)

AIS_CSV = os.path.join(
    BASE_DIR,
    "data",
    "ais",
    "sar_vessel_detections_pipev4_20260901.csv"
)

# ============================================================
# SENTINEL-1 PATCH OFFSET
# ============================================================

PATCH_OFFSET_X = 3524
PATCH_OFFSET_Y = 10429


# ============================================================
# SAR ACQUISITION TIME
# ============================================================

SAR_TIMESTAMP = "2026-09-01T04:55:50.331759Z"


# ============================================================
# LOAD GCPs
# ============================================================

def load_gcps(xml_path):

    root = ET.parse(
        xml_path
    ).getroot()

    gcps = []

    for point in root.iter():

        if (
            point.tag.split("}")[-1]
            != "geolocationGridPoint"
        ):
            continue

        values = {}

        for child in point:

            name = child.tag.split("}")[-1]

            if child.text is not None:

                values[name] = child.text

        if all(
            key in values
            for key in [
                "line",
                "pixel",
                "latitude",
                "longitude"
            ]
        ):

            gcps.append(
                GroundControlPoint(

                    row=float(
                        values["line"]
                    ),

                    col=float(
                        values["pixel"]
                    ),

                    x=float(
                        values["longitude"]
                    ),

                    y=float(
                        values["latitude"]
                    )
                )
            )

    return gcps


# ============================================================
# PIXEL -> LAT/LON
# ============================================================

def pixel_to_latlon(
    transformer,
    full_x,
    full_y
):

    lon, lat = transformer.xy(
        full_y,
        full_x
    )

    return (
        float(lat),
        float(lon)
    )


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk(
    confidence,
    status,
    distance_km
):

    risk = 0

    if status == "Dark Ship":

        risk += 60

    elif status == "AIS Blackout":

        risk += 40

    else:

        risk += 10

    if (
        distance_km is not None
    ):

        if distance_km > 10:

            risk += 20

        elif distance_km > 5:

            risk += 10

        elif distance_km > 2:

            risk += 5

    if confidence < 0.20:

        risk += 15

    elif confidence < 0.40:

        risk += 10

    elif confidence < 0.60:

        risk += 5

    return min(
        risk,
        100
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [
        MODEL_PATH,
        IMAGE_PATH,
        XML_PATH,
        AIS_CSV
    ]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nFile not found:\n{path}"
            )

    # --------------------------------------------------------
    # GCPs
    # --------------------------------------------------------

    gcps = load_gcps(
        XML_PATH
    )

    print(
        "GCPs loaded:",
        len(gcps)
    )

    transformer = GCPTransformer(
        gcps
    )

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    print(
        "Loading YOLO model..."
    )

    model = YOLO(
        MODEL_PATH
    )

    print(
        "Running ship detection..."
    )

    results = model.predict(
        source=IMAGE_PATH,
        conf=0.01,
        verbose=False
    )

    result = results[0]

    boxes = result.boxes

    print()
    print(
        "===== DARK SHIP DETECTION ====="
    )

    if boxes is None or len(boxes) == 0:

        print(
            "No ships detected."
        )

        transformer.close()

        return

    # --------------------------------------------------------
    # Process ships
    # --------------------------------------------------------

    for i, box in enumerate(
        boxes,
        1
    ):

        xyxy = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        x1, y1, x2, y2 = xyxy

        confidence = float(
            box.conf[0]
            .cpu()
            .numpy()
        )

        # Patch center
        patch_x = (
            x1 + x2
        ) / 2

        patch_y = (
            y1 + y2
        ) / 2

        # Full Sentinel-1 pixel
        full_x = (
            patch_x
            + PATCH_OFFSET_X
        )

        full_y = (
            patch_y
            + PATCH_OFFSET_Y
        )

        # Geographic coordinates
        latitude, longitude = (
            pixel_to_latlon(
                transformer,
                full_x,
                full_y
            )
        )

        # ----------------------------------------------------
        # SPATIAL + TEMPORAL AIS MATCH
        # ----------------------------------------------------

        ais = match_ship(

            latitude,

            longitude,

            SAR_TIMESTAMP,

            AIS_CSV
        )

        status = ais[
            "status"
        ]

        distance = ais[
            "distance_km"
        ]

        time_difference = ais[
            "time_difference_min"
        ]

        mmsi = ais[
            "mmsi"
        ]

        category = ais[
            "category"
        ]

        matching_score = ais[
            "matching_score"
        ]

        # Risk
        risk_score = calculate_risk(

            confidence,

            status,

            distance
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        print()
        print(
            f"SHIP {i}"
        )

        print(
            f"Confidence: "
            f"{confidence:.4f}"
        )

        print(
            f"Patch Center: "
            f"({patch_x:.1f}, "
            f"{patch_y:.1f})"
        )

        print(
            f"Full Pixel: "
            f"({full_x:.1f}, "
            f"{full_y:.1f})"
        )

        print(
            f"Latitude: "
            f"{latitude:.6f}"
        )

        print(
            f"Longitude: "
            f"{longitude:.6f}"
        )

        print(
            f"Status: "
            f"{status}"
        )

        print(
            "AIS Distance: "
            f"{distance:.2f} km"
            if distance is not None
            else
            "AIS Distance: N/A"
        )

        print(
            "AIS Time Difference: "
            f"{time_difference:.2f} min"
            if time_difference is not None
            else
            "AIS Time Difference: N/A"
        )

        print(
            f"MMSI: "
            f"{mmsi}"
            if mmsi is not None
            else
            "MMSI: N/A"
        )

        print(
            f"Category: "
            f"{category}"
        )

        print(
            "Matching Score: "
            f"{matching_score}"
            if matching_score is not None
            else
            "Matching Score: N/A"
        )

        print(
            f"Risk Score: "
            f"{risk_score}/100"
        )

    transformer.close()

    print()
    print(
        "===== PIPELINE COMPLETE ====="
    )


if __name__ == "__main__":

    main()