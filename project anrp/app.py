
from pathlib import Path
import hashlib

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR

from src.anpr_pipeline import (
    MODEL_PATH,
    IMAGE_SIZE,
    DETECTION_CONFIDENCE,
    expand_box,
    prepare_crop,
    extract_paddle_results,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="PlateVision | ANPR Dashboard",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0b1120;
        --panel: #111b2e;
        --border: #24334b;
        --text: #edf4ff;
        --muted: #91a4bf;
        --accent: #40dfc0;
        --blue: #72a7ff;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 15% 0%, #142943 0%, transparent 38%),
            #0b1120;
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #0d1728;
        border-right: 1px solid #24334b;
    }

    [data-testid="stSidebar"] * {
        color: #dce8f7;
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #edf4ff !important;
        letter-spacing: -0.5px;
    }

    p, label, .stMarkdown {
        color: #c3d0e2;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0 0 30px 0;
    }

    .brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 46px;
        height: 46px;
        border-radius: 14px;
        background: linear-gradient(135deg, #40dfc0, #62a6ff);
        color: #081321;
        font-size: 25px;
        font-weight: 700;
    }

    .brand-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: #f4f8ff;
    }

    .brand-caption {
        font-size: 11px;
        letter-spacing: 2px;
        color: #8da3bf;
        text-transform: uppercase;
    }

    .hero {
        padding: 30px 32px;
        margin-bottom: 24px;
        border: 1px solid #293b54;
        border-radius: 22px;
        background: linear-gradient(
            115deg,
            rgba(22, 45, 68, 0.98),
            rgba(17, 27, 46, 0.96) 65%,
            rgba(19, 54, 64, 0.8)
        );
    }

    .eyebrow {
        color: #40dfc0;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2.2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero h1 {
        font-size: clamp(28px, 4vw, 43px);
        line-height: 1.15;
        margin: 0 0 12px 0;
    }

    .hero p {
        max-width: 700px;
        color: #aabbd1;
        font-size: 15px;
        line-height: 1.8;
        margin-bottom: 18px;
    }

    .pill {
        display: inline-block;
        border: 1px solid #316b68;
        background: rgba(64, 223, 192, 0.08);
        color: #6cebd2;
        border-radius: 100px;
        padding: 6px 12px;
        margin: 0 7px 5px 0;
        font-size: 11px;
        font-weight: 600;
    }

    .section-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 20px;
        font-weight: 600;
        color: #edf4ff;
        margin: 28px 0 5px 0;
    }

    .section-subtitle {
        font-size: 13px;
        color: #91a4bf;
        margin-bottom: 17px;
    }

    .panel {
        background: rgba(17, 27, 46, 0.9);
        border: 1px solid #24334b;
        border-radius: 17px;
        padding: 20px;
        margin-bottom: 14px;
    }

    .metric-label {
        font-size: 12px;
        color: #91a4bf;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #f2f7ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 29px;
        font-weight: 700;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: #40dfc0;
        font-size: 11px;
        margin-top: 7px;
    }

    .plate-card {
        border: 1px solid #2d776e;
        background: linear-gradient(135deg, #132c38, #102335);
        padding: 24px;
        border-radius: 18px;
        text-align: center;
        margin: 10px 0 20px 0;
    }

    .plate-label {
        color: #78bdb6;
        font-size: 11px;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .plate-number {
        display: inline-block;
        color: #f2fff9;
        background: #101c2a;
        border: 2px solid #40dfc0;
        border-radius: 9px;
        padding: 12px 20px;
        font-family: 'Space Grotesk', monospace;
        font-size: clamp(22px, 3vw, 34px);
        font-weight: 700;
        letter-spacing: 3px;
        overflow-wrap: anywhere;
    }

    .muted {
        color: #91a4bf;
        font-size: 12px;
    }

    .stButton > button {
        border: 1px solid #40dfc0;
        border-radius: 10px;
        min-height: 47px;
        background: linear-gradient(100deg, #40dfc0, #70e6d0);
        color: #081923;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #85f5df;
        background: #85f5df;
        color: #081923;
        transform: translateY(-1px);
    }

    .stDownloadButton > button {
        border: 1px solid #34516e;
        background: #17263b;
        color: #dce8f7;
        border-radius: 10px;
        min-height: 43px;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed #42617d;
        border-radius: 14px;
        padding: 10px;
        background: rgba(18, 34, 54, 0.65);
    }

    [data-testid="stMetric"] {
        background: #111b2e;
        border: 1px solid #24334b;
        padding: 17px;
        border-radius: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #91a4bf;
    }

    [data-testid="stMetricValue"] {
        color: #edf4ff;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #24334b;
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: #24334b;
    }

    .footer {
        text-align: center;
        padding-top: 24px;
        color: #657c98;
        font-size: 11px;
        letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD MODELS ONCE AND REUSE THEM
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained YOLO model not found: {MODEL_PATH}"
        )

    detector = YOLO(str(MODEL_PATH))

    reader = PaddleOCR(
        lang="en",
        use_textline_orientation=False,
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    return detector, reader


# --------------------------------------------------
# IMAGE ANALYSIS
# --------------------------------------------------
def analyze_image(image_rgb, detector, reader):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]

    prediction = detector.predict(
        source=image_bgr,
        imgsz=IMAGE_SIZE,
        conf=DETECTION_CONFIDENCE,
        device="cpu",
        verbose=False,
    )[0]

    annotated = image_bgr.copy()
    plates = []

    if prediction.boxes is not None:
        for box in prediction.boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            x1, y1, x2, y2 = expand_box(
                x1, y1, x2, y2, width, height
            )

            crop = image_bgr[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            plate_text = ""
            ocr_confidence = 0.0
            status = "No text recognized"

            try:
                ocr_image = prepare_crop(crop)
                ocr_results = reader.predict(input=ocr_image)

                plate_text, ocr_confidence = (
                    extract_paddle_results(ocr_results)
                )

                if plate_text:
                    status = "Text recognized"

            except Exception:
                status = "OCR error"

            detection_confidence = float(box.conf[0].item())

            # Draw the detection box and readable label.
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (64, 223, 192),
                3,
            )

            label = plate_text or "Plate detected"
            label_y = max(24, y1 - 10)

            cv2.putText(
                annotated,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (64, 223, 192),
                2,
                cv2.LINE_AA,
            )

            plates.append({
                "Plate number": plate_text or "Not recognized",
                "Detection confidence": round(
                    detection_confidence * 100, 2
                ),
                "OCR confidence": round(
                    float(ocr_confidence) * 100, 2
                ),
                "Bounding box": (
                    f"({x1}, {y1}, {x2}, {y2})"
                ),
                "Status": status,
            })

    plates.sort(
        key=lambda item: item["Detection confidence"],
        reverse=True,
    )

    annotated_rgb = cv2.cvtColor(
        annotated, cv2.COLOR_BGR2RGB
    )

    return annotated_rgb, plates


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">P</div>
            <div>
                <div class="brand-name">PlateVision</div>
                <div class="brand-caption">ANPR Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Navigation")
    st.markdown("🖼️ &nbsp; Image analysis")
    st.markdown("📊 &nbsp; Recognition results")
    st.markdown("⚙️ &nbsp; Model information")

    st.divider()

    st.markdown("#### SYSTEM STATUS")
    st.markdown(
        '<span class="pill">● IMAGE MODE</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="muted">
            <b>Detector</b><br>YOLO<br><br>
            <b>Text recognition</b><br>PaddleOCR<br><br>
            <b>Execution</b><br>Local CPU inference
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("Master's project · Computer Vision & AI")


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">COMPUTER VISION / INTELLIGENT RECOGNITION</div>
        <h1>Automatic Number Plate Recognition</h1>
        <p>
            Turn vehicle images into structured information.
            Detect license plates with YOLO, extract their text
            using PaddleOCR, and inspect the results in one workspace.
        </p>
        <span class="pill">YOLO DETECTION</span>
        <span class="pill">OCR RECOGNITION</span>
        <span class="pill">IMAGE PROCESSING</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
st.markdown(
    '<div class="section-heading">01 / Image workspace</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">Upload a vehicle image to start the analysis.</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Supported formats: JPG, JPEG, PNG and BMP.",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="panel">
            <div style="font-size:34px;margin-bottom:10px;">⌁</div>
            <h3>Ready to analyze</h3>
            <p class="muted">
                Upload a vehicle image above. The system will detect
                license plates and attempt to recognize the text.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.pop("anpr_result", None)
    st.session_state.pop("anpr_image_key", None)

else:
    uploaded_bytes = uploaded_file.getvalue()
    image_key = hashlib.sha256(uploaded_bytes).hexdigest()

    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(pil_image)
    except Exception:
        st.error("This file could not be opened as a valid image.")
        st.stop()

    # Do not display results belonging to a previous upload.
    if st.session_state.get("anpr_image_key") != image_key:
        st.session_state.pop("anpr_result", None)
        st.session_state["anpr_image_key"] = image_key

    info_col, info_col2 = st.columns([3, 1])

    with info_col:
        st.markdown(
            f"**Selected image:** `{uploaded_file.name}`"
        )

    with info_col2:
        st.markdown(
            f'<div class="muted" style="text-align:right;">'
            f'{image_rgb.shape[1]} × {image_rgb.shape[0]} pixels'
            f'</div>',
            unsafe_allow_html=True,
        )

    preview_col, action_col = st.columns([1.35, 1])

    with preview_col:
        st.image(
            image_rgb,
            caption="Original image",
            use_container_width=True,
        )

    with action_col:
        st.markdown(
            """
            <div class="panel">
                <div class="eyebrow">ANALYSIS PIPELINE</div>
                <h3>Ready for detection</h3>
                <p class="muted">
                    1. Locate the plate with YOLO<br><br>
                    2. Prepare the detected plate crop<br><br>
                    3. Extract text with PaddleOCR<br><br>
                    4. Review the results and confidence scores
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        run_analysis = st.button(
            "✦  Analyze image",
            type="primary",
            use_container_width=True,
        )

        if run_analysis:
            try:
                with st.spinner(
                    "Running YOLO detection and OCR..."
                ):
                    detector, reader = load_models()
                    annotated_rgb, plates = analyze_image(
                        image_rgb, detector, reader
                    )

                st.session_state["anpr_result"] = {
                    "image_key": image_key,
                    "annotated": annotated_rgb,
                    "plates": plates,
                }

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------
    result = st.session_state.get("anpr_result")

    if result and result["image_key"] == image_key:
        st.divider()

        st.markdown(
            '<div class="section-heading">02 / Analysis results</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtitle">Detection output and OCR summary for your image.</div>',
            unsafe_allow_html=True,
        )

        plates = result["plates"]
        annotated_rgb = result["annotated"]

        detected_count = len(plates)
        recognized = [
            p for p in plates
            if p["Status"] == "Text recognized"
        ]

        if plates:
            mean_detection = np.mean([
                p["Detection confidence"] for p in plates
            ])
        else:
            mean_detection = 0.0

        if recognized:
            mean_ocr = np.mean([
                p["OCR confidence"] for p in recognized
            ])
        else:
            mean_ocr = 0.0

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                "Plates detected",
                detected_count,
                help="Number of license plate detections.",
            )

        with m2:
            st.metric(
                "Text recognized",
                len(recognized),
                help="Detections where OCR returned text.",
            )

        with m3:
            st.metric(
                "Mean detection confidence",
                f"{mean_detection:.2f}%",
            )

        with m4:
            st.metric(
                "Mean OCR confidence",
                f"{mean_ocr:.2f}%",
            )

        original_col, output_col = st.columns(2, gap="large")

        with original_col:
            st.markdown("#### Original")
            st.image(
                image_rgb,
                use_container_width=True,
            )

        with output_col:
            st.markdown("#### YOLO detection output")
            st.image(
                annotated_rgb,
                use_container_width=True,
            )

        st.markdown(
            '<div class="section-heading">03 / Recognized plate details</div>',
            unsafe_allow_html=True,
        )

        if plates:
            for index, plate in enumerate(plates, start=1):
                plate_text = plate["Plate number"]

                if plate_text != "Not recognized":
                    st.markdown(
                        f"""
                        <div class="plate-card">
                            <div class="plate-label">
                                DETECTED LICENSE PLATE · {index:02d}
                            </div>
                            <div class="plate-number">
                                {plate_text}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(
                        f"Plate {index}: detected, but text was not recognized."
                    )

            results_df = pd.DataFrame(plates)

            st.markdown("#### Detection details")
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Detection confidence": st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                    "OCR confidence": st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                },
            )

            csv_data = results_df.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "⬇  Export results as CSV",
                data=csv_data,
                file_name="platevision_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        else:
            st.warning(
                "No license plates were detected in this image. "
                "Try an image with a clearer, more visible plate."
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        PLATEVISION · YOLO + PADDLEOCR · IMAGE-BASED ANPR
        <br><br>
        Built for academic demonstration and computer vision evaluation.
    </div>
    """,
    unsafe_allow_html=True,
)

