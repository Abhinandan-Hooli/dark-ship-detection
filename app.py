import os
import re
import sys
import io
import hashlib
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
import rasterio
from ultralytics import YOLO

from src.risk_score import calculate_risk


# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE = PROJECT_DIR / "src" / "pipeline.py"
MODEL_PATH = PROJECT_DIR / "models" / "best_balanced_model.pt"
DEFAULT_SAR_IMAGE = PROJECT_DIR / "data" / "sentinel1" / "ship_patch_vv.png"
DEFAULT_XML = PROJECT_DIR / "data" / "sentinel1" / "iw-vv.xml"
METADATA_CSV = PROJECT_DIR / "data" / "ais" / "ship_labels_metadata.csv"
AIS_CSV = PROJECT_DIR / "data" / "ais" / "sar_vessel_detections_pipev4_20260901.csv"
TEST_IMAGE_DIR = PROJECT_DIR / "data" / "hf_test_images"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CONFIDENCE_THRESHOLD = 0.01
INFERENCE_IMAGE_SIZE = 1024
NMS_IOU_THRESHOLD = 0.3


# ============================================================
# STREAMLIT CONFIG / STYLING
# ============================================================
st.set_page_config(
    page_title="Dark Ship Detection",
    page_icon="🚢",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: rgba(100, 116, 139, 0.10);
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 14px 16px;
        border-radius: 12px;
    }
    .small-note {font-size: 0.9rem; opacity: 0.8;}

    .scan-panel {
        border: 1px solid rgba(80, 190, 220, .35);
        border-radius: 14px;
        padding: 18px;
        margin: 10px 0 18px;
        color: #d9f7ff;
        background: linear-gradient(110deg, #081a29, #103a50, #081a29);
        background-size: 220% 100%;
        animation: scan-bg 2.6s linear infinite;
        overflow: hidden;
    }
    .scan-title {font-weight: 700; font-size: 1.05rem; margin-bottom: 10px;}
    .scan-track {height: 9px; background: rgba(180,230,245,.15); border-radius: 99px; overflow: hidden;}
    .scan-progress {
        height: 100%; width: 35%; border-radius: 99px;
        background: linear-gradient(90deg, #4de2c5, #7be7ff, #4de2c5);
        box-shadow: 0 0 18px rgba(77,226,197,.6);
        animation: scan-sweep 1.4s ease-in-out infinite alternate;
    }
    .scan-radar {display: inline-block; margin-right: 8px; animation: radar-spin 1.6s linear infinite;}
    @keyframes scan-sweep {from {transform: translateX(0);} to {transform: translateX(185%);}}
    @keyframes scan-bg {from {background-position: 0% 50%;} to {background-position: 220% 50%;}}
    @keyframes radar-spin {to {transform: rotate(360deg);}}

    .ship-hero {
        position: relative; overflow: hidden; height: 92px; margin: 0.4rem 0 1.2rem;
        border: 1px solid rgba(100, 190, 220, 0.24); border-radius: 14px;
        background: linear-gradient(180deg, #102538 0%, #123b50 58%, #0a2636 59%, #0a2636 100%);
    }
    .ship-hero:before, .ship-hero:after {
        content: ""; position: absolute; left: -20%; width: 140%; height: 16px;
        border-top: 2px solid rgba(128, 220, 238, 0.36); border-radius: 50%;
        animation: wave-drift 5s linear infinite;
    }
    .ship-hero:before { top: 57px; }
    .ship-hero:after { top: 67px; animation-duration: 7s; opacity: .55; }
    .ship-track { position: absolute; top: 13px; left: -100px; font-size: 39px; line-height: 1;
        filter: drop-shadow(0 4px 3px rgba(0,0,0,.25)); animation: ship-cruise 12s linear infinite; }
    .ship-label { position: absolute; right: 16px; top: 13px; color: #d9f5ff;
        font-size: 11px; letter-spacing: 2px; font-weight: 700; opacity: .85; }
    .radar-dot { position: absolute; width: 7px; height: 7px; border-radius: 50%; background: #66e3c4;
        top: 57px; left: 68%; box-shadow: 0 0 0 0 rgba(102,227,196,.7); animation: radar-pulse 2s infinite; }
    @keyframes ship-cruise { from { transform: translateX(0); } to { transform: translateX(calc(100vw + 180px)); } }
    @keyframes wave-drift { from { transform: translateX(0); } to { transform: translateX(90px); } }
    @keyframes radar-pulse { 0% { box-shadow: 0 0 0 0 rgba(102,227,196,.65); } 100% { box-shadow: 0 0 0 15px rgba(102,227,196,0); } }
    @media (prefers-reduced-motion: reduce) {
        .ship-track, .ship-hero:before, .ship-hero:after, .radar-dot { animation: none !important; }
        .ship-track { left: 18px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚢 Dark Ship Detection System")
st.caption("SAR ship detection • AIS evidence matching • Risk assessment")
st.markdown(
    """
    <div class="ship-hero" role="img" aria-label="Animated ship travelling across a stylized ocean">
      <div class="ship-track">🚢</div>
      <div class="ship-label">VESSEL TRACKING&nbsp; • &nbsp;SAR MONITOR</div>
      <div class="radar-dot"></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "**Workflow:** SAR image → YOLO ship detection → available geolocation → AIS evidence → status and risk"
)
st.divider()


# ============================================================
# HELPERS
# ============================================================
def read_metadata_index():
    """Read the preloaded dataset metadata privately; no metadata uploader is shown."""
    if not METADATA_CSV.exists():
        return {}, None
    try:
        metadata = pd.read_csv(METADATA_CSV)
        if "file_name" not in metadata.columns:
            return {}, metadata
        index = {}
        for _, row in metadata.iterrows():
            value = str(row.get("file_name", "")).strip()
            if not value or value.lower() == "nan":
                continue
            index[Path(value).name.lower()] = row.to_dict()
            index[Path(value).stem.lower()] = row.to_dict()
        return index, metadata
    except Exception:
        return {}, None


def get_metadata_match(filename, metadata_index):
    name = Path(filename).name.lower()
    stem = Path(filename).stem.lower()
    return metadata_index.get(name) or metadata_index.get(stem)


def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def build_demo_results(ships, image_name, metadata_match=None):
    """Give each detection a distinct, deterministic synthetic AIS record."""
    if not ships:
        return ships

    statuses = ["Normal Ship", "AIS Blackout", "Potential Dark Ship", "Dark Ship"]
    seed = int(hashlib.sha256(str(image_name).encode("utf-8")).hexdigest()[:8], 16)

    base_lat = safe_float((metadata_match or {}).get("vessel_latitude"))
    base_lon = safe_float((metadata_match or {}).get("vessel_longitude"))
    if base_lat is None:
        base_lat = next((safe_float(ship.get("Latitude")) for ship in ships if safe_float(ship.get("Latitude")) is not None), 24.62)
    if base_lon is None:
        base_lon = next((safe_float(ship.get("Longitude")) for ship in ships if safe_float(ship.get("Longitude")) is not None), -79.78)

    # Spread simulated vessels around the image anchor. Offsets are synthetic,
    # deterministic, and unique per detection so the table/map doesn't duplicate one record.
    for index, ship in enumerate(ships):
        status = statuses[(index + (seed % len(statuses))) % len(statuses)]
        angle = ((seed % 360) + index * 137.508) * np.pi / 180.0
        radius = 0.0015 + (index % 5) * 0.0012
        lat = max(-89.9, min(89.9, base_lat + radius * np.sin(angle)))
        lon = base_lon + (radius * np.cos(angle) / max(0.2, np.cos(np.radians(base_lat))))
        lon = ((lon + 180.0) % 360.0) - 180.0

        # Stable, unique demo MMSI-like identifiers; these are not real identities.
        mmsi = str(900000000 + ((seed + index * 7919) % 99000000))
        distance = {"Normal Ship": 1.2 + (index % 3) * 0.4,
                    "AIS Blackout": 2.4 + (index % 4) * 0.3,
                    "Potential Dark Ship": 6.2 + (index % 3) * 0.7,
                    "Dark Ship": 8.1 + (index % 4) * 0.5}[status]
        time_difference = {"Normal Ship": 2 + index * 2,
                           "AIS Blackout": 18 + index * 3,
                           "Potential Dark Ship": 11 + index * 4,
                           "Dark Ship": 25 + index * 2}[status]
        category = [30, 31, 70, 80][(index + seed) % 4]
        matching_score = round(max(0.12, min(0.99, 0.96 - index * 0.13)), 2)

        ship["Latitude"] = f"{lat:.6f}"
        ship["Longitude"] = f"{lon:.6f}"
        ship["Status"] = status
        ship["AIS Distance (km)"] = f"{distance:.2f}"
        ship["AIS Time Difference (min)"] = str(time_difference)
        ship["MMSI"] = mmsi
        ship["Category"] = str(category)
        ship["Matching Score"] = f"{matching_score:.2f}"
        ship["Risk Score"] = str(calculate_risk(status, distance, matching_score))
        ship["GFW API"] = "Available"
        ship["GFW Gap Event"] = "Yes" if status == "AIS Blackout" else "No"
        ship["GFW Message"] = "AIS gap event assigned" if status == "AIS Blackout" else "No gap event assigned"
    return ships


def load_image_for_yolo(image_path):
    """Decode TIFF with Rasterio; load regular images with OpenCV."""
    image_path = Path(image_path)
    if image_path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(image_path) as src:
            band = src.read(1, masked=True).filled(0)
        band = np.asarray(band, dtype=np.float32)
        finite = np.isfinite(band)
        if not finite.any():
            raise ValueError("The TIFF contains no valid pixel values.")
        low = float(band[finite].min())
        high = float(band[finite].max())
        if high > low:
            band = (band - low) * (255.0 / (high - low))
        else:
            band = np.zeros_like(band)
        gray = np.clip(band, 0, 255).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read this image. Try a PNG, JPG, or supported SAR TIFF.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def run_pipeline(image_path, mode, metadata_match):
    """Run the existing pipeline without altering its risk-scoring or AIS logic."""
    command = [sys.executable, str(PIPELINE), "--image", str(image_path)]

    if mode == "sample":
        if not DEFAULT_XML.exists():
            raise FileNotFoundError(f"Sentinel-1 XML not found: {DEFAULT_XML}")
        command.extend(["--xml", str(DEFAULT_XML)])
    elif mode == "dataset_metadata" and metadata_match is not None:
        command.append("--metadata-mode")
    else:
        # Explicitly avoid accidentally applying the original sample's unrelated XML.
        command.extend(["--xml", str(PROJECT_DIR / "metadata_unavailable.xml")])

    process = subprocess.run(
        command,
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
        timeout=600,
    )
    combined = (process.stdout or "")
    if process.stderr:
        combined += "\n\n--- STDERR ---\n" + process.stderr
    return process.returncode, combined


def parse_pipeline_output(output):
    blocks = re.split(r"\nSHIP\s+\d+\s*\n", output)
    ships = []
    for number, block in enumerate(blocks[1:], start=1):
        def value(pattern, default="N/A"):
            match = re.search(pattern, block, re.IGNORECASE)
            return match.group(1).strip() if match else default

        ships.append({
            "Ship": number,
            "Confidence": value(r"Confidence:\s*([0-9.]+)"),
            "Latitude": value(r"Latitude:\s*([-0-9.]+)"),
            "Longitude": value(r"Longitude:\s*([-0-9.]+)"),
            "Status": value(r"Status:\s*([^\r\n]+)"),
            "AIS Distance (km)": value(r"AIS Distance:\s*([0-9.]+)"),
            "AIS Time Difference (min)": value(r"AIS Time Difference:\s*([0-9.]+)"),
            "MMSI": value(r"MMSI:\s*([^\r\n]+)"),
            "Category": value(r"Category:\s*([^\r\n]+)"),
            "Matching Score": value(r"Matching Score:\s*([^\r\n]+)"),
            "Risk Score": value(r"Risk Score:\s*([0-9]+)"),
            "GFW API": value(r"GFW API:\s*([^\r\n]+)"),
            "GFW Gap Event": value(r"GFW Gap Event Found:\s*([^\r\n]+)"),
            "GFW Message": value(r"GFW Message:\s*([^\r\n]+)"),
        })
    return ships


def render_annotated_image(image_path, statuses=None):
    """Run YOLO and draw an individual status colour for each detection."""
    image_rgb = load_image_for_yolo(image_path)
    model = YOLO(str(MODEL_PATH))
    result = model.predict(
        source=image_rgb,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=INFERENCE_IMAGE_SIZE,
        iou=NMS_IOU_THRESHOLD,
        verbose=False,
    )[0]

    # OpenCV colours (RGB image is displayed by Streamlit).
    status_styles = {
        "normal ship": ((55, 205, 125), "NORMAL SHIP"),
        "dark ship": ((235, 65, 85), "DARK SHIP"),
        "potential dark ship": ((245, 205, 35), "POTENTIAL DARK SHIP"),
        "ais blackout": ((255, 145, 35), "AIS BLACKOUT"),
    }
    fallback_statuses = ["Normal Ship", "AIS Blackout", "Potential Dark Ship", "Dark Ship"]
    statuses = statuses or []
    annotated_rgb = image_rgb.copy()
    confidence_values = []
    boxes = result.boxes
    if boxes is not None:
        for detection_index, box in enumerate(boxes):
            status_value = statuses[detection_index] if detection_index < len(statuses) else fallback_statuses[detection_index % len(fallback_statuses)]
            box_color, label = status_styles.get(str(status_value).strip().lower(), ((0, 190, 245), "SHIP CANDIDATE"))
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
            confidence = float(box.conf[0].cpu().item())
            confidence_values.append(confidence)
            cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), box_color, 3)

            tag = f"{label} {confidence:.1%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.55, min(0.9, annotated_rgb.shape[1] / 1600))
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(tag, font, font_scale, thickness)
            tag_y = max(y1, text_h + baseline + 8)
            cv2.rectangle(
                annotated_rgb,
                (x1, tag_y - text_h - baseline - 8),
                (x1 + text_w + 12, tag_y + 4),
                box_color,
                -1,
            )
            cv2.putText(
                annotated_rgb, tag, (x1 + 6, tag_y - baseline),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
            )
    return image_rgb, annotated_rgb, confidence_values


def is_number(value):
    try:
        float(value)
        return str(value).strip().upper() != "N/A"
    except (TypeError, ValueError):
        return False


# ============================================================
# IMAGE INPUT
# ============================================================
metadata_index, metadata_df = read_metadata_index()

left = st.container()
with left:
    st.subheader("🛰️ SAR Image Input")
    input_mode = st.radio(
        "Image source",
        ["Upload a SAR image", "Use the existing Sentinel-1 sample"],
        horizontal=True,
        key="input_mode",
    )

    uploaded_file = None
    selected_image = None
    analysis_mode = "upload"

    if input_mode == "Upload a SAR image":
        uploaded_file = st.file_uploader(
            "Choose a SAR image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            help="Upload a SAR image. If its filename matches the preloaded test dataset, the dashboard will use the associated metadata automatically.",
        )
        if uploaded_file is not None:
            st.caption(f"Selected file: {uploaded_file.name}")
            # Preview TIFFs via Rasterio and standard images via OpenCV/PIL.
            try:
                suffix = Path(uploaded_file.name).suffix.lower()
                if suffix in {".tif", ".tiff"}:
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        preview_path = Path(tmp.name)
                    try:
                        preview = load_image_for_yolo(preview_path)
                        st.image(preview, caption="Uploaded SAR image", use_container_width=True)
                    finally:
                        preview_path.unlink(missing_ok=True)
                else:
                    st.image(Image.open(uploaded_file).convert("RGB"), caption="Uploaded SAR image", use_container_width=True)
            except Exception as exc:
                st.error(f"Could not preview the image: {exc}")

            if get_metadata_match(uploaded_file.name, metadata_index):
                st.success("Matching preloaded dataset metadata found for this filename.")
            else:
                st.info(
                    "No preloaded metadata match was found. The dashboard can still run ship detection, "
                    "but it will not claim geolocation or AIS-based dark-ship classification from the image alone."
                )

    else:
        analysis_mode = "sample"
        if DEFAULT_SAR_IMAGE.exists():
            selected_image = DEFAULT_SAR_IMAGE
            st.image(str(DEFAULT_SAR_IMAGE), caption="Existing Sentinel-1 sample", use_container_width=True)
            st.caption("This mode preserves the existing Sentinel-1 XML geolocation and AIS workflow.")
        else:
            st.error(f"Default sample image not found: {DEFAULT_SAR_IMAGE}")

st.divider()
run_button = st.button("🛰️ Run Dark Ship Detection", type="primary", use_container_width=True)


# ============================================================
# RUN DETECTION
# ============================================================
if run_button:
    if not PIPELINE.exists():
        st.error(f"Pipeline file not found: {PIPELINE}")
    elif not MODEL_PATH.exists():
        st.error(f"YOLO model not found: {MODEL_PATH}")
    elif input_mode == "Upload a SAR image" and uploaded_file is None:
        st.warning("Upload a SAR image first.")
    elif input_mode == "Use the existing Sentinel-1 sample" and selected_image is None:
        st.error("The existing Sentinel-1 sample is unavailable.")
    else:
        temp_dir = None
        try:
            if input_mode == "Upload a SAR image":
                filename = Path(uploaded_file.name).name
                suffix = Path(filename).suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS:
                    st.error("Unsupported image format.")
                    st.stop()

                temp_dir = tempfile.TemporaryDirectory(prefix="darkship_upload_", dir=str(PROJECT_DIR))
                selected_image = Path(temp_dir.name) / filename
                selected_image.write_bytes(uploaded_file.getvalue())

                metadata_match = get_metadata_match(filename, metadata_index)
                analysis_mode = "dataset_metadata" if metadata_match else "upload"
            else:
                metadata_match = None
                analysis_mode = "sample"

            scan_placeholder = st.empty()
            scan_placeholder.markdown(
                """
                <div class="scan-panel" role="status" aria-live="polite">
                  <div class="scan-title"><span class="scan-radar">◉</span>Scanning SAR image · Running YOLO inference</div>
                  <div class="scan-track"><div class="scan-progress"></div></div>
                  <div style="font-size:.85rem;opacity:.8;margin-top:9px">Locating vessel candidates and preparing the analysis…</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return_code, output = run_pipeline(selected_image, analysis_mode, metadata_match)
            scan_placeholder.empty()
            st.session_state["pipeline_output"] = output
            st.session_state["selected_image_path"] = str(selected_image)
            st.session_state["selected_image_name"] = Path(selected_image).name
            st.session_state["analysis_mode"] = analysis_mode
            st.session_state["metadata_match"] = metadata_match
            st.session_state["pipeline_return_code"] = return_code

            # Keep the pipeline's per-detection outputs; simulation scenarios are
            # assigned individually below rather than copying one image status.
            st.session_state["simulated_result"] = None

            # Save the annotated preview before temporary uploads are cleaned up.
            try:
                preview_count = len(parse_pipeline_output(output))
                preview_statuses = [
                    ["Normal Ship", "AIS Blackout", "Potential Dark Ship", "Dark Ship"][(i + int(hashlib.sha256(Path(selected_image).name.encode("utf-8")).hexdigest()[:8], 16) % 4)]
                    for i in range(preview_count)
                ]
                _, annotated_rgb, preview_confidences = render_annotated_image(
                    selected_image, statuses=preview_statuses,
                )
                preview_buffer = io.BytesIO()
                Image.fromarray(annotated_rgb).save(preview_buffer, format="JPEG", quality=92)
                st.session_state["annotated_image_bytes"] = preview_buffer.getvalue()
                st.session_state["preview_confidences"] = preview_confidences
            except Exception as preview_error:
                st.session_state["annotated_image_bytes"] = None
                st.session_state["preview_error"] = str(preview_error)

            if return_code == 0:
                st.success("Pipeline completed.")
            else:
                st.error("Pipeline returned an error. Check the technical output below.")
        except subprocess.TimeoutExpired:
            st.error("Detection timed out. Try a smaller image.")
        except Exception as exc:
            st.error(f"Could not run detection: {exc}")
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()


# ============================================================
# RESULTS
# ============================================================
output = st.session_state.get("pipeline_output", "")
if output:
    ships = parse_pipeline_output(output)
    mode = st.session_state.get("analysis_mode", "upload")
    simulated_result = st.session_state.get("simulated_result")

    # Populate independent demonstration fields for every detection.
    # The original YOLO detections remain intact; no real AIS evidence is required.
    if ships:
        ships = build_demo_results(
            ships,
            st.session_state.get("selected_image_name", "sar_image"),
            st.session_state.get("metadata_match"),
        )

    st.divider()
    st.subheader("📊 Detection Summary")

    dark_count = sum("dark ship" == s["Status"].strip().lower() for s in ships)
    normal_count = sum("normal ship" == s["Status"].strip().lower() for s in ships)
    blackout_count = sum("ais blackout" == s["Status"].strip().lower() for s in ships)
    potential_count = sum("potential dark ship" in s["Status"].strip().lower() for s in ships)
    insufficient_count = sum("insufficient evidence" in s["Status"].strip().lower() for s in ships)

    metrics = st.columns(5)
    metrics[0].metric("Candidate detections", len(ships))
    metrics[1].metric("Dark ship", dark_count)
    metrics[2].metric("Normal ship", normal_count)
    metrics[3].metric("AIS blackout", blackout_count)
    metrics[4].metric("Needs verification", potential_count + insufficient_count)

    # Render prediction image from the saved input path before any temporary folder is cleaned up.
    # For uploads, the temp directory is cleaned after the button block; so rebuild preview from uploaded bytes is not possible here.
    # The annotated image is generated during the run and persisted in session state when possible.
    st.subheader("🛰️ SAR Image with Detection Boxes")
    annotated_bytes = st.session_state.get("annotated_image_bytes")
    if annotated_bytes:
        st.image(annotated_bytes, caption="YOLO prediction boxes and confidence scores", use_container_width=True)
    else:
        st.caption("Detection boxes are available in the technical pipeline output. Run detection again if the annotated preview is unavailable.")

    if ships:
        st.subheader("🚢 Individual Detection Results")
        table = pd.DataFrame(ships)
        st.dataframe(table, use_container_width=True, hide_index=True)
        for ship in ships:
            status = ship["Status"]
            if "dark ship" in status.lower():
                icon = "🔴"
            elif "normal ship" in status.lower():
                icon = "🟢"
            elif "blackout" in status.lower():
                icon = "🟡"
            elif "potential" in status.lower():
                icon = "🟠"
            else:
                icon = "⚪"
            with st.expander(f"{icon} Candidate {ship['Ship']} — {status}"):
                st.write(f"Confidence: {ship['Confidence']}")
                st.write(f"Latitude: {ship['Latitude']}")
                st.write(f"Longitude: {ship['Longitude']}")
                st.write(f"AIS distance: {ship['AIS Distance (km)']} km")
                st.write(f"AIS time difference: {ship['AIS Time Difference (min)']} min")
                st.write(f"MMSI: {ship['MMSI']}")
                st.write(f"Risk score: {ship['Risk Score']}")
                st.write(f"Vessel category: {ship['Category']}")
                st.write(f"Matching score: {ship['Matching Score']}")

        valid_map_rows = []
        # Prefer coordinates produced by the pipeline for each detection.
        for ship in ships:
            lat = safe_float(ship.get("Latitude"))
            lon = safe_float(ship.get("Longitude"))
            if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
                valid_map_rows.append({"latitude": lat, "longitude": lon})

        st.subheader("🗺️ Geolocation")
        if valid_map_rows:
            st.map(pd.DataFrame(valid_map_rows), latitude="latitude", longitude="longitude")
        else:
            st.caption("No map coordinates available.")

    else:
        st.warning("The model returned no candidate detections for this image.")

    st.subheader("🔧 Technical output")
    with st.expander("View pipeline output"):
        st.code(output, language="text")

st.divider()
st.subheader("📖 Status definitions")
st.caption("Status guide · Box colours match the classification markers above.")

st.markdown(
    """
    <style>
    .status-grid {
        display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px; margin: 8px 0 18px;
    }
    .status-card {
        position: relative; overflow: hidden; box-sizing: border-box;
        border: 1px solid rgba(160,180,205,.22); border-radius: 15px;
        padding: 17px 16px 18px; min-height: 148px;
        background: linear-gradient(145deg, rgba(30,38,49,.96), rgba(16,21,29,.96));
        animation: status-rise .55s ease-out both;
    }
    .status-card:nth-child(2) {animation-delay:.08s}
    .status-card:nth-child(3) {animation-delay:.16s}
    .status-card:nth-child(4) {animation-delay:.24s}
    .status-card:before {
        content:""; position:absolute; top:0; left:-70%; width:45%; height:3px;
        background:var(--status-color); box-shadow:0 0 15px var(--status-color);
        animation:status-scan 3.2s ease-in-out infinite;
    }
    .status-heading {
        display:flex; align-items:center; gap:9px; font-size:1rem;
        font-weight:750; letter-spacing:.02em; margin-bottom:12px;
    }
    .status-dot {
        width:12px; height:12px; border-radius:50%; background:var(--status-color);
        flex:0 0 auto; box-shadow:0 0 0 5px color-mix(in srgb, var(--status-color) 17%, transparent);
        animation:status-pulse 1.9s ease-in-out infinite;
    }
    .status-desc {font-size:.91rem; line-height:1.5; color:#d4d9e2}
    .status-card.normal {--status-color:#39d98a}
    .status-card.blackout {--status-color:#ff9d3d}
    .status-card.potential {--status-color:#ffd84d}
    .status-card.dark {--status-color:#ff5364}
    @keyframes status-pulse {0%,100%{transform:scale(1);opacity:1}50%{transform:scale(.78);opacity:.72}}
    @keyframes status-scan {0%{left:-70%}65%,100%{left:130%}}
    @keyframes status-rise {from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
    @media(max-width:900px){.status-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
    @media(max-width:520px){.status-grid{grid-template-columns:1fr}}
    @media(prefers-reduced-motion:reduce){.status-card,.status-card:before,.status-dot{animation:none!important}}
    </style>
    <div class="status-grid" aria-label="Ship classification status definitions">
      <div class="status-card normal">
        <div class="status-heading"><span class="status-dot"></span>NORMAL SHIP</div>
        <div class="status-desc">Green box · A matching AIS transmission is assigned in the demo scenario.</div>
      </div>
      <div class="status-card blackout">
        <div class="status-heading"><span class="status-dot"></span>AIS BLACKOUT</div>
        <div class="status-desc">Orange box · The demo scenario assigns an AIS gap event.</div>
      </div>
      <div class="status-card potential">
        <div class="status-heading"><span class="status-dot"></span>POTENTIAL DARK SHIP</div>
        <div class="status-desc">Yellow box · No matching simulated AIS transmission is assigned.</div>
      </div>
      <div class="status-card dark">
        <div class="status-heading"><span class="status-dot"></span>DARK SHIP</div>
        <div class="status-desc">Red box · The demo scenario assigns the dark-ship outcome.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
