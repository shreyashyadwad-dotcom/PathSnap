# PathSnap – GPS Map Matching and Route Reconstruction

## 1. Project Overview

PathSnap is a GPS map-matching and route reconstruction system that converts noisy GPS observations into realistic road-network trajectories.

The system combines Machine Learning, Hidden Markov Models (HMM) and the Viterbi algorithm to identify the most likely sequence of road segments followed by a vehicle.

### Key Features

- GPS trajectory visualization using Folium.
- Candidate road-segment generation.
- ML-based candidate probability prediction.
- HMM-based emission and transition probability computation.
- Viterbi decoding for optimal route reconstruction.
- FastAPI backend for processing reconstruction requests.
- Streamlit frontend for interactive visualization.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Model Evaluation | Cross-validation, OOF predictions |
| Sequence Modelling | Hidden Markov Model |
| Decoding | Viterbi Algorithm |
| Backend | FastAPI |
| Frontend | Streamlit |
| Map Visualization | Folium |
| Data Processing | Pandas, NumPy |
| API Communication | REST API |

## 3. Project Structure

```text
PathSnap/
│
├── artifacts/
│   ├── pathsnap_best_model.joblib
│   ├── pathsnap_model_metadata.json
│   ├── student_test_candidate_predictions.csv
│   └── pathsnap_tuned_model_comparison.csv
│
├── data/
│   ├── student_test/
│   ├── processed/
│   └── road_network/
│
├── frontend/
│   └── app.py
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── config.py
│   │   ├── model_loader.py
│   │   └── prediction_service.py
│   │
│   ├── data/
│   ├── decoder/
│   ├── models/
│   └── services/
│
├── tests/
│
├── pathsnap_model_training_colab.ipynb
├── requirements.txt
└── README.md
```

*The structure above illustrates the main project components. Update it to match the final repository.*

## 4. Installation

### Prerequisites

- Python 3.10 or later (use a version compatible with the project dependencies).
- Git.
- Required GPS datasets and road-network data.

### Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd PathSnap
```

### Create a Virtual Environment

Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Model Training

The model training workflow is provided in:

```text
pathsnap_model_training_colab.ipynb
```

The notebook performs:

1. Dataset loading and preprocessing.
2. Feature preparation.
3. Model training.
4. Cross-validation and out-of-fold evaluation.
5. Hyperparameter tuning.
6. Model comparison.
7. Best-model selection.
8. Model and evaluation artifact export.

The selected model and supporting artifacts are saved under the `artifacts/` directory.

Run the notebook using Jupyter or VS Code.

**Note:** If the trained model artifacts are already supplied, retraining is not required to run inference.

## 6. Running the Application

PathSnap uses two separate services: FastAPI for the backend and Streamlit for the frontend.

### Terminal 1 – Start the Backend

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Backend:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

### Terminal 2 – Start the Frontend

```bash
python -m streamlit run frontend/app.py --server.port 8501
```

Frontend:

```text
http://localhost:8501
```

Keep both terminals running while using the application.

## 7. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Backend health and model status |
| GET | `/trips` | Retrieve available GPS trips |
| POST | `/reconstruct` | Perform ML + HMM + Viterbi reconstruction |
| GET | `/results/{trip_id}` | Retrieve reconstruction results |

Additional prediction and artifact-status endpoints may be available depending on the integrated backend version.

## 8. Application Workflow

1. Start the FastAPI backend.
2. Start the Streamlit frontend.
3. Open the application in a browser.
4. Select an available GPS trip.
5. View its raw GPS trajectory on the interactive map.
6. Request route reconstruction.
7. The backend processes the observations using the ML and HMM pipeline.
8. The reconstructed route results are returned through the API.

## 9. Testing

Run the project test suite using:

```bash
pytest -q
```

Run individual test files using:

```bash
pytest tests/test_transitions.py -v
```

## 10. Deployment and Configuration

- Configure the backend API address in `frontend/app.py`.
- Ensure the required GPS datasets and road-network files are available.
- Place the trained model and metadata in the expected artifact locations.
- Configure network access and firewall permissions when accessing the application from another machine.
- Do not commit passwords, API keys, private credentials or other secrets.

For LAN access, the backend can be started with:

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## 11. Supporting Results

The repository should include relevant supporting materials, where applicable:

- Trained model weights.
- Model metadata and configuration files.
- Tuned model comparison results.
- Candidate prediction CSV files.
- Experiment and training logs.
- Visualization scripts.
- Test cases and evaluation results.

## 12. Project Objective

PathSnap aims to improve GPS trajectory reconstruction by combining learned candidate probabilities with spatial and temporal continuity constraints, producing more realistic road-matched vehicle trajectories.
