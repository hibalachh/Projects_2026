
# PlateVision — Automatic Number Plate Recognition (ANPR)

PlateVision is an image-based Automatic Number Plate Recognition system that detects vehicle license plates using YOLO and recognizes plate text using PaddleOCR. It includes a Streamlit dashboard for interactive use and a FastAPI service for programmatic access.

## Features

- License plate detection using a trained YOLO model.
- License plate text recognition using PaddleOCR.
- Automatic plate cropping and image preprocessing.
- Annotated images with detection bounding boxes.
- Detection and OCR confidence scores.
- Interactive Streamlit dashboard.
- CSV export of recognition results.
- FastAPI endpoints with interactive Swagger documentation.

> **Current scope:** Image processing only. Video processing and temporal tracking are not implemented.

## Tech Stack

- **Language:** Python
- **Object detection:** Ultralytics YOLO
- **OCR:** PaddleOCR / PaddlePaddle
- **Image processing:** OpenCV, NumPy, Pillow
- **Frontend:** Streamlit
- **API:** FastAPI, Uvicorn
- **Data handling:** Pandas

## Project Structure

```text
ANPR-From-Scratch/
├── app.py
├── src/
│   ├── api.py
│   ├── anpr_pipeline.py
│   └── prepare_yolo_dataset.py
├── data/
│   └── processed/
│       └── license_plates/
│           ├── dataset.yaml
│           ├── images/
│           └── labels/
├── outputs/
│   ├── runs/
│   │   └── license_plate_detector/
│   │       └── weights/
│   │           └── best.pt
│   └── anpr/
├── requirements.txt
└── README.md
```

Some generated folders may be absent until the dataset preparation or model inference scripts are run.

## Installation

### 1. Requirements

- Python 3.13 (development environment)
- pip
- Internet access for installing dependencies and downloading OCR models when needed

CPU inference is supported by the current configuration.

### 2. Activate the virtual environment

From the project root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If the virtual environment already exists, simply activate it.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Streamlit Dashboard

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

Upload a JPG, JPEG, PNG, or BMP image, then select **Analyze image** to view the detection and recognition results.

## Run the FastAPI Service

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

API endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service information |
| `/health` | GET | Model loading status |
| `/analyze` | POST | Analyze an uploaded image |
| `/docs` | GET | Interactive API documentation |

Open `http://127.0.0.1:8000/docs` to upload a test image through Swagger UI and inspect the JSON response.

Run Streamlit and FastAPI in separate terminals if you want both services active at once.

## Model and Dataset

The application expects the trained YOLO weights at:

`outputs/runs/license_plate_detector/weights/best.pt`

The prepared dataset configuration is expected at:

`data/processed/license_plates/dataset.yaml`

Ensure the model weights and any required dataset files are available before running the application. Large model weights and datasets may need to be distributed separately from the source code.

## Evaluation Results

The current reviewed test set produced the following results:

- **Detection coverage on images marked as containing a visible plate:** 100%
- **OCR exact-match accuracy on reviewed, labeled visible plates:** 30/36 (83.33%)
- **Mean Character Error Rate (CER):** 4.51%

These results apply to the reviewed test subset and are not a guarantee of performance on unseen real-world images. Six visible images were missing ground-truth labels and were excluded from the exact-match calculation.

## Limitations

- Accuracy depends on image quality, lighting, viewing angle, and plate visibility.
- OCR confidence is a model score, not a guarantee of correctness.
- CPU inference may be slower than GPU inference.
- The current system processes images and does not support video tracking.
- Validate recognition results before using them in operational decisions.

## Future Improvements

- Deploy the Streamlit dashboard online.
- Improve OCR performance on challenging images.
- Add more diverse, labeled training and evaluation images.
- Add automated testing, logging, and model monitoring.
- Improve error handling and API security before public deployment.

## License

No project license has been specified yet. Add an appropriate `LICENSE` file before public distribution, and verify the licensing terms of the dataset and model weights.

## Author

Master's project focused on computer vision, object detection, and optical character recognition.
