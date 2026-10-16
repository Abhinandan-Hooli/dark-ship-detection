import os
import numpy as np
import rasterio
import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_balanced_model.pt")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "hf_test_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "visualizations")

TEST_IMAGES = [
    "lat02442838lon-7945241__time20221110.tif",
    "lat02462130lon-7978721__time20221110.tif",
    "lat02480483lon-7970996__time20200804.tif",
    "lat02493796lon-7960086__time20200804.tif",
    "lat02510766lon-7967182__time20231209.tif",
]

CONFIDENCE_THRESHOLD = 0.01


def load_tiff_as_rgb(image_path):
    """Read a single-band SAR TIFF and scale it the same way as pipeline.py."""
    with rasterio.open(image_path) as src:
        image_array = src.read(1, masked=True).filled(0)

    image_array = np.asarray(image_array, dtype=np.float32)
    finite = np.isfinite(image_array)
    if not finite.any():
        raise ValueError(f"No valid pixel values found in {image_path}")

    low = float(image_array[finite].min())
    high = float(image_array[finite].max())
    if high > low:
        image_array = (image_array - low) * (255.0 / (high - low))
    else:
        image_array = np.zeros_like(image_array)

    gray = np.clip(image_array, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, np.newaxis], 3, axis=2)


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)

    print("Generating YOLO visualization images...")
    print(f"Model: {MODEL_PATH}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}")

    for filename in TEST_IMAGES:
        image_path = os.path.join(IMAGE_DIR, filename)
        if not os.path.isfile(image_path):
            print(f"\nSKIPPED (file not found): {filename}")
            continue

        try:
            image_rgb = load_tiff_as_rgb(image_path)
            result = model.predict(
                source=image_rgb,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )[0]

            # Ultralytics result.plot() returns a BGR image suitable for cv2.imwrite.
            annotated_bgr = result.plot()
            output_name = os.path.splitext(filename)[0] + "_detections.jpg"
            output_path = os.path.join(OUTPUT_DIR, output_name)

            if not cv2.imwrite(output_path, annotated_bgr):
                raise OSError(f"Could not save output image: {output_path}")

            boxes = result.boxes
            count = 0 if boxes is None else len(boxes)
            confidences = [] if boxes is None else [
                float(value) for value in boxes.conf.cpu().numpy()
            ]

            print(f"\nImage: {filename}")
            print(f"Candidate boxes: {count}")
            if confidences:
                print("Confidence scores: " + ", ".join(f"{v:.4f}" for v in confidences))
            print(f"Saved: {output_path}")

        except Exception as exc:
            print(f"\nERROR processing {filename}: {exc}")

    print("\nDone. Open outputs\\visualizations to inspect the bounding boxes.")
    print("Note: boxes are model predictions, not confirmed ship labels.")


if __name__ == "__main__":
    main()
