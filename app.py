import streamlit as st
import subprocess
import sys
import os
import re
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINE = os.path.join(
    PROJECT_DIR,
    "src",
    "pipeline.py"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "best_balanced_model.pt"
)

SAR_IMAGE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "sentinel1",
    "ship_patch_vv.png"
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dark Ship Detection",
    page_icon="🚢",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🚢 Dark Ship Detection System")

st.caption(
    "SAR-based ship detection with AIS matching and risk assessment"
)

st.divider()


# ============================================================
# SYSTEM PIPELINE
# ============================================================

st.subheader("System Pipeline")

st.markdown(
    """
    **SAR Image → YOLOv8 → Geolocation → AIS Matching → Risk Score**
    """
)

st.divider()


# ============================================================
# RUN BUTTON
# ============================================================

if st.button(
    "🔍 Run Dark Ship Detection",
    type="primary",
    use_container_width=False
):

    with st.spinner("Running complete detection pipeline..."):

        try:

            process = subprocess.run(
                [
                    sys.executable,
                    PIPELINE
                ],
                capture_output=True,
                text=True,
                cwd=PROJECT_DIR
            )

            output = process.stdout
            error_output = process.stderr

            st.session_state["pipeline_output"] = output

            if process.returncode != 0:

                st.error("Pipeline failed.")

                if error_output:
                    st.code(
                        error_output,
                        language="text"
                    )

            else:

                st.success(
                    "Detection pipeline completed successfully."
                )

        except Exception as e:

            st.error(
                f"Could not run pipeline: {e}"
            )


# ============================================================
# GET PIPELINE OUTPUT
# ============================================================

output = st.session_state.get(
    "pipeline_output",
    ""
)


# ============================================================
# PARSE SHIP RESULTS
# ============================================================

ships = []


if output:

    blocks = re.split(
        r"\nSHIP\s+\d+\s*\n",
        output
    )

    ship_number = 0

    for block in blocks[1:]:

        ship_number += 1

        def get_value(pattern, default="N/A"):

            match = re.search(
                pattern,
                block,
                re.IGNORECASE
            )

            if match:
                return match.group(1).strip()

            return default


        confidence = get_value(
            r"Confidence:\s*([0-9.]+)"
        )

        patch_center = get_value(
            r"Patch Center:\s*\(([^)]+)\)"
        )

        full_pixel = get_value(
            r"Full Pixel:\s*\(([^)]+)\)"
        )

        latitude = get_value(
            r"Latitude:\s*([-0-9.]+)"
        )

        longitude = get_value(
            r"Longitude:\s*([-0-9.]+)"
        )

        status = get_value(
            r"Status:\s*(.+)"
        )

        ais_distance = get_value(
            r"AIS Distance:\s*([0-9.]+)"
        )

        ais_time = get_value(
            r"AIS Time Difference:\s*([0-9.]+)"
        )

        mmsi = get_value(
            r"MMSI:\s*(.+)"
        )

        category = get_value(
            r"Category:\s*(.+)"
        )

        matching_score = get_value(
            r"Matching Score:\s*(.+)"
        )

        risk_score = get_value(
            r"Risk Score:\s*([0-9]+)"
        )

        ships.append(
            {
                "ship": ship_number,
                "confidence": confidence,
                "patch_center": patch_center,
                "full_pixel": full_pixel,
                "latitude": latitude,
                "longitude": longitude,
                "status": status,
                "ais_distance": ais_distance,
                "ais_time": ais_time,
                "mmsi": mmsi,
                "category": category,
                "matching_score": matching_score,
                "risk_score": risk_score
            }
        )


# ============================================================
# SUMMARY
# ============================================================

if ships:

    dark_ships = sum(
        1 for s in ships
        if "Dark Ship" in s["status"]
    )

    normal_ships = sum(
        1 for s in ships
        if "Normal Ship" in s["status"]
    )

    ais_blackout = sum(
        1 for s in ships
        if "AIS Blackout" in s["status"]
    )

    st.subheader("📊 Detection Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Ships Detected",
        len(ships)
    )

    c2.metric(
        "🔴 Dark Ships",
        dark_ships
    )

    c3.metric(
        "🟢 Normal Ships",
        normal_ships
    )

    c4.metric(
        "🟡 AIS Blackout",
        ais_blackout
    )

    st.divider()


    # ========================================================
    # SAR IMAGE + YOLO DETECTIONS
    # ========================================================

    st.subheader("🛰️ SAR Image & YOLO Detection")

    if os.path.exists(SAR_IMAGE):

        try:

            image = Image.open(
                SAR_IMAGE
            ).convert("RGB")

            # ------------------------------------------------
            # RUN YOLO DIRECTLY FOR VISUALIZATION
            # ------------------------------------------------

            try:

                from ultralytics import YOLO

                model = YOLO(
                    MODEL_PATH
                )

                results = model.predict(
                    source=SAR_IMAGE,
                    conf=0.01,
                    verbose=False
                )

                result = results[0]

                draw = ImageDraw.Draw(
                    image
                )

                # Try to load a font
                try:
                    font = ImageFont.truetype(
                        "arial.ttf",
                        18
                    )
                except:
                    font = ImageFont.load_default()


                if result.boxes is not None:

                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()

                    for idx, (
                        box,
                        conf
                    ) in enumerate(
                        zip(
                            boxes,
                            confidences
                        ),
                        1
                    ):

                        x1, y1, x2, y2 = box

                        x1 = int(x1)
                        y1 = int(y1)
                        x2 = int(x2)
                        y2 = int(y2)

                        # Draw bounding box
                        draw.rectangle(
                            [
                                x1,
                                y1,
                                x2,
                                y2
                            ],
                            outline="red",
                            width=4
                        )

                        label = (
                            f"Ship {idx} "
                            f"{conf:.3f}"
                        )

                        # Text background
                        bbox = draw.textbbox(
                            (x1, y1),
                            label,
                            font=font
                        )

                        draw.rectangle(
                            bbox,
                            fill="red"
                        )

                        draw.text(
                            (
                                x1,
                                y1
                            ),
                            label,
                            fill="white",
                            font=font
                        )


                st.image(
                    image,
                    caption="Sentinel-1 SAR image with YOLO ship detections",
                    use_container_width=True
                )

            except Exception as e:

                st.warning(
                    "Could not draw YOLO boxes."
                )

                st.image(
                    image,
                    caption="Sentinel-1 SAR image",
                    use_container_width=True
                )

                st.code(
                    str(e)
                )

        except Exception as e:

            st.error(
                f"Could not open SAR image: {e}"
            )

    else:

        st.warning(
            f"SAR image not found:\n{SAR_IMAGE}"
        )


    st.divider()


    # ========================================================
    # SHIP DETECTION RESULTS
    # ========================================================

    st.subheader("🚢 Ship Detection Results")


    for ship in ships:

        status = ship["status"]

        if "Dark Ship" in status:

            icon = "🔴"

        elif "Normal Ship" in status:

            icon = "🟢"

        elif "AIS Blackout" in status:

            icon = "🟡"

        else:

            icon = "⚪"


        st.markdown(
            f"### {icon} SHIP {ship['ship']} — {status.upper()}"
        )


        c1, c2, c3 = st.columns(3)


        with c1:

            st.markdown("**YOLO Confidence**")
            st.write(
                ship["confidence"]
            )

            st.markdown("**Latitude**")
            st.write(
                ship["latitude"]
            )

            st.markdown("**Longitude**")
            st.write(
                ship["longitude"]
            )


        with c2:

            st.markdown("**AIS Distance**")
            st.write(
                f"{ship['ais_distance']} km"
            )

            st.markdown("**AIS Time Difference**")
            st.write(
                f"{ship['ais_time']} min"
            )

            st.markdown("**MMSI**")
            st.write(
                ship["mmsi"]
            )


        with c3:

            st.markdown("**Category**")
            st.write(
                ship["category"]
            )

            st.markdown("**Matching Score**")
            st.write(
                ship["matching_score"]
            )

            st.markdown("**Risk Score**")
            st.write(
                f"{ship['risk_score']}/100"
            )


        st.divider()


    # ========================================================
    # MAP
    # ========================================================

    st.subheader("🗺️ Detected Ship Locations")

    map_data = []

    for ship in ships:

        try:

            lat = float(
                ship["latitude"]
            )

            lon = float(
                ship["longitude"]
            )

            map_data.append(
                {
                    "latitude": lat,
                    "longitude": lon
                }
            )

        except:

            pass


    if map_data:

        map_df = pd.DataFrame(
            map_data
        )

        st.map(
            map_df,
            latitude="latitude",
            longitude="longitude",
            zoom=5
        )

    else:

        st.info(
            "No valid coordinates available for map."
        )


    # ========================================================
    # TECHNICAL OUTPUT
    # ========================================================

    st.subheader(
        "🔧 Technical Pipeline Output"
    )

    with st.expander(
        "View Technical Pipeline Output"
    ):

        st.code(
            output,
            language="text"
        )


else:

    st.info(
        "Click **Run Dark Ship Detection** to start the analysis."
    )


# ============================================================
# STATUS DEFINITIONS
# ============================================================

st.divider()

st.subheader("📖 Status Definitions")

c1, c2, c3 = st.columns(3)


with c1:

    st.success("🟢 NORMAL SHIP")

    st.write(
        "SAR detection has a valid spatial and temporal AIS match."
    )


with c2:

    st.warning("🟡 AIS BLACKOUT")

    st.write(
        "AIS information is temporarily unavailable or stale."
    )


with c3:

    st.error("🔴 DARK SHIP")

    st.write(
        "SAR detection has no valid AIS match within the defined spatial and temporal criteria."
    )