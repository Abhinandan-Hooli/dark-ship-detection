import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import rasterio
import numpy as np
from rasterio.control import GroundControlPoint
from rasterio.transform import GCPTransformer
from ultralytics import YOLO

from ais_match import match_ship
from risk_score import calculate_risk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_balanced_model.pt")
DEFAULT_IMAGE_PATH = os.path.join(BASE_DIR, "data", "sentinel1", "ship_patch_vv.png")
DEFAULT_XML_PATH = os.path.join(BASE_DIR, "data", "sentinel1", "iw-vv.xml")
AIS_CSV = os.path.join(BASE_DIR, "data", "ais", "sar_vessel_detections_pipev4_20260901.csv")
METADATA_CSV = os.path.join(BASE_DIR, "data", "ais", "ship_labels_metadata.csv")
DEFAULT_PATCH_OFFSET_X = 3524
DEFAULT_PATCH_OFFSET_Y = 10429
DEFAULT_SAR_TIMESTAMP = "2026-09-01T04:55:50.331759Z"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dark Ship Detection Pipeline")
    parser.add_argument("--image", type=str, default=None, help="Path to SAR image")
    parser.add_argument("--xml", type=str, default=None, help="Sentinel-1 geolocation XML")
    parser.add_argument("--timestamp", type=str, default=None, help="SAR acquisition timestamp")
    parser.add_argument("--offset-x", type=float, default=None, help="Sentinel-1 patch X offset")
    parser.add_argument("--offset-y", type=float, default=None, help="Sentinel-1 patch Y offset")
    parser.add_argument("--metadata-mode", action="store_true", help="Use matching row from ship_labels_metadata.csv for a dataset TIFF")
    return parser.parse_args()


def load_gcps(xml_path):
    root = ET.parse(xml_path).getroot()
    gcps = []
    for point in root.iter():
        if point.tag.split("}")[-1] != "geolocationGridPoint":
            continue
        values = {child.tag.split("}")[-1]: child.text for child in point if child.text is not None}
        required = ("line", "pixel", "latitude", "longitude")
        if all(key in values for key in required):
            gcps.append(GroundControlPoint(row=float(values["line"]), col=float(values["pixel"]), x=float(values["longitude"]), y=float(values["latitude"])))
    return gcps


def pixel_to_latlon(transformer, full_x, full_y):
    lon, lat = transformer.xy(full_y, full_x)
    return float(lat), float(lon)


def find_metadata_row(image_path):
    if not os.path.exists(METADATA_CSV):
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")
    metadata = pd.read_csv(METADATA_CSV)
    required = {"file_name", "timestamp_s1", "vessel_latitude", "vessel_longitude"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError("Metadata CSV missing required columns: " + ", ".join(sorted(missing)))

    # Dataset metadata stores the scene identifier without the .tif extension.
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    rows = metadata[metadata["file_name"].astype(str).str.strip() == image_stem]
    if rows.empty:
        raise ValueError(f"No metadata row matches image filename '{image_stem}' in {METADATA_CSV}")
    if len(rows) > 1:
        # A filename can have multiple vessel annotations; use first row for scene-level metadata,
        # but report the count so this cannot silently appear to be a unique record.
        print(f"Warning: {len(rows)} metadata rows match this image; using the first row for scene metadata.")
    return rows.iloc[0]


def safe_value(row, column):
    if column not in row.index or pd.isna(row[column]):
        return None
    return row[column]


def main():
    args = parse_arguments()
    image_path = args.image or DEFAULT_IMAGE_PATH
    xml_path = args.xml if args.xml is not None else DEFAULT_XML_PATH
    sar_timestamp = args.timestamp or DEFAULT_SAR_TIMESTAMP
    patch_offset_x = args.offset_x if args.offset_x is not None else DEFAULT_PATCH_OFFSET_X
    patch_offset_y = args.offset_y if args.offset_y is not None else DEFAULT_PATCH_OFFSET_Y

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    metadata_row = None
    metadata_mode = args.metadata_mode
    if metadata_mode:
        metadata_row = find_metadata_row(image_path)
        raw_timestamp = safe_value(metadata_row, "timestamp_s1")
        if raw_timestamp is not None:
            # This dataset stores timestamp_s1 as Unix seconds.
            sar_timestamp = pd.to_datetime(float(raw_timestamp), unit="s", utc=True).isoformat()
        xml_path = None  # Do not accidentally apply the unrelated original sample XML.

    geolocation_available = bool(not metadata_mode and xml_path and os.path.exists(xml_path))
    transformer = None
    if geolocation_available:
        gcps = load_gcps(xml_path)
        print("GCPs loaded:", len(gcps))
        if gcps:
            transformer = GCPTransformer(gcps)
        else:
            geolocation_available = False

    print("Analysis Mode:", "Dataset Metadata + YOLO Detection (AIS verification not assumed)" if metadata_mode else ("Full Geolocation + AIS Analysis" if geolocation_available else "YOLO Detection Only"))
    print("Image:", image_path)
    print("SAR Timestamp:", sar_timestamp)

    if metadata_mode:
        print("Metadata file:", METADATA_CSV)
        for col, label in (("vessel_name", "Metadata vessel name"), ("vessel_mmsi", "Metadata MMSI"), ("vessel_type", "Metadata vessel type"), ("vessel_latitude", "Metadata latitude"), ("vessel_longitude", "Metadata longitude")):
            print(f"{label}: {safe_value(metadata_row, col)}")
        print("Note: dataset metadata is not proof that the vessel is a dark ship.")

    model = YOLO(MODEL_PATH)

    # OpenCV/Ultralytics may fail to read some single-band SAR GeoTIFFs.
    # Read TIFFs with rasterio, scale the band to 8-bit, and pass an array to YOLO.
    if os.path.splitext(image_path)[1].lower() in {".tif", ".tiff"}:
        with rasterio.open(image_path) as src:
            image_array = src.read(1, masked=True).filled(0)
        image_array = np.asarray(image_array, dtype=np.float32)
        finite = np.isfinite(image_array)
        if not finite.any():
            raise ValueError(f"TIFF contains no valid pixel values: {image_path}")
        low = float(image_array[finite].min())
        high = float(image_array[finite].max())
        if high > low:
            image_array = (image_array - low) * (255.0 / (high - low))
        else:
            image_array = np.zeros_like(image_array)
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        image_array = np.repeat(image_array[:, :, np.newaxis], 3, axis=2)
        results = model.predict(source=image_array, conf=0.01, verbose=False)
    else:
        results = model.predict(source=image_path, conf=0.01, verbose=False)
    result = results[0]
    boxes = result.boxes
    print("\n===== DARK SHIP DETECTION =====")
    if boxes is None or len(boxes) == 0:
        print("No ships detected by the current YOLO model.")
        if transformer is not None:
            transformer.close()
        return

    for i, box in enumerate(boxes, 1):
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        confidence = float(box.conf[0].cpu().numpy())
        patch_x, patch_y = (x1 + x2) / 2, (y1 + y2) / 2
        latitude = longitude = None
        status = "AIS Analysis Unavailable"
        distance = time_difference = mmsi = matching_score = risk_score = None
        category = "N/A"

        if metadata_mode:
            # These are scene/label coordinates from the dataset metadata, not a pixel-derived
            # coordinate for every detection. Do not claim precise per-detection geolocation.
            latitude = safe_value(metadata_row, "vessel_latitude")
            longitude = safe_value(metadata_row, "vessel_longitude")
            mmsi = safe_value(metadata_row, "vessel_mmsi")
            category = safe_value(metadata_row, "vessel_type") or "N/A"
            status = "Insufficient Evidence — AIS Verification Required"
        elif geolocation_available and transformer is not None:
            full_x = patch_x + patch_offset_x
            full_y = patch_y + patch_offset_y
            latitude, longitude = pixel_to_latlon(transformer, full_x, full_y)
            ais = match_ship(latitude, longitude, sar_timestamp, AIS_CSV)
            status = ais.get("status", "AIS Analysis Unavailable")
            distance = ais.get("distance_km")
            time_difference = ais.get("time_difference_min")
            mmsi = ais.get("mmsi")
            category = ais.get("category", "N/A")
            matching_score = ais.get("matching_score")
            if distance is not None:
                risk_score = calculate_risk(status, distance, matching_score)

        print(f"\nSHIP {i}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Detection center: ({patch_x:.1f}, {patch_y:.1f})")
        print(f"Latitude: {latitude if latitude is not None else 'N/A'}")
        print(f"Longitude: {longitude if longitude is not None else 'N/A'}")
        print(f"Status: {status}")
        print(f"AIS Distance: {distance if distance is not None else 'N/A'}")
        print(f"AIS Time Difference: {time_difference if time_difference is not None else 'N/A'}")
        print(f"MMSI: {mmsi if mmsi is not None else 'N/A'}")
        print(f"Category: {category}")
        print(f"Matching Score: {matching_score if matching_score is not None else 'N/A'}")
        print(f"Risk Score: {str(risk_score) + '/100' if risk_score is not None else 'N/A'}")
        if metadata_mode:
            print("Alert: No dark-ship alarm generated from metadata alone; matching historical AIS evidence is required.")

    if transformer is not None:
        transformer.close()
    print("\n===== PIPELINE COMPLETE =====")


if __name__ == "__main__":
    main()
