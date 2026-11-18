
"""FastAPI backend for PathSnap map-matching and route reconstruction."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.api.schemas import (
    HealthResponse,
    ReconstructRequest,
    ReconstructResponse,
    TripSummary,
    TripsListResponse,
)
from src.api.model_loader import get_artifact_status
from src.api.prediction_service import PredictionService
from src.data.loader import load_trip
from src.models.classifier import get_classifier
from src.services.reconstruction import reconstruct_trip, _GRAPH_CACHE
from src.api.config import STUDENT_TEST_DIR


# ==================================================
# 1. LOGGING CONFIGURATION
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("pathsnap.api")


# ==================================================
# 2. GLOBAL SERVICES AND RESULT CACHE
# ==================================================

_STORED_RESULTS: dict[str, dict[str, Any]] = {}

prediction_service = PredictionService()


# ==================================================
# 3. PREDICTION REQUEST VALIDATION
# ==================================================

class PredictionRequest(BaseModel):
    trip_id: str = Field(
        pattern=r"^\d{8}$",
        description="Eight-digit student-testing trip ID",
    )


# ==================================================
# 4. APPLICATION LIFESPAN
# ==================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle manager."""

    logger.info("Initializing PathSnap backend service...")

    classifier = get_classifier()

    if classifier.is_loaded:
        logger.info(
            "Candidate classifier loaded: %s",
            classifier.model_path,
        )
    else:
        logger.warning(
            "No classifier loaded at startup; will attempt lazy loading."
        )

    yield

    logger.info("Shutting down PathSnap backend service.")


# ==================================================
# 5. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="PathSnap API",
    description=(
        "Distributed ML & HMM/Viterbi Map-Matching "
        "and Route Reconstruction Engine"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ==================================================
# 6. CORS CONFIGURATION
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# 7. HEALTH ENDPOINT
# ==================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """Check health, model status, and road graph status."""

    classifier = get_classifier()

    data_dir = Path("data") / "student_test"

    trip_count = (
        len([d for d in data_dir.iterdir() if d.is_dir()])
        if data_dir.is_dir()
        else 0
    )

    return HealthResponse(
        status="healthy",
        model_loaded=classifier.is_loaded,
        model_name=classifier.metadata.get(
            "model_name",
            "Random Forest Classifier"
            if classifier.is_loaded
            else "Heuristic Fallback",
        ),
        road_graph_cached=len(_GRAPH_CACHE) > 0,
        available_trips_count=trip_count,
        version="1.0.0",
    )


# ==================================================
# 8. TRIP DISCOVERY ENDPOINT
# ==================================================

@app.get("/trips", response_model=TripsListResponse, tags=["Trips"])
@app.get("/trips/", response_model=TripsListResponse, tags=["Trips"])
def list_trips(split: str = "student_test") -> TripsListResponse:
    """Return available trips from the selected dataset split."""

    data_path = Path("data") / split

    if not data_path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset split folder not found: {data_path}",
        )

    folders = sorted(
        d.name for d in data_path.iterdir() if d.is_dir()
    )

    summaries: list[TripSummary] = []

    for trip_id in folders:
        trip_dir = data_path / trip_id

        track_file = trip_dir / f"{trip_id}.track"
        route_file = trip_dir / f"{trip_id}.route"

        total_obs = 0

        if track_file.is_file():
            try:
                with open(track_file, "r", encoding="utf-8") as f:
                    total_obs = sum(1 for _ in f)
            except Exception:
                logger.exception(
                    "Could not count observations for trip %s",
                    trip_id,
                )

        summaries.append(
            TripSummary(
                trip_id=trip_id,
                split=split,
                total_observations=total_obs,
                has_ground_truth=route_file.is_file(),
                label=f"Evaluator Trip {trip_id} ({total_obs} GPS pts)",
            )
        )

    logger.info("Discovered %d trips in split %s", len(summaries), split)

    return TripsListResponse(
        trips=summaries,
        total_trips=len(summaries),
    )


# ==================================================
# 9. HMM + ML ROUTE RECONSTRUCTION
# ==================================================

@app.post(
    "/reconstruct",
    response_model=ReconstructResponse,
    tags=["Reconstruction"],
)
def reconstruct(req: ReconstructRequest) -> ReconstructResponse:
    """Run full HMM + ML Viterbi route reconstruction."""

    trip_id = req.trip_id.strip()
    split = req.split.strip()

    try:
        classifier = get_classifier()

        result = reconstruct_trip(
            trip_id=trip_id,
            split=split,
            data_dir="data",
            classifier=classifier,
        )

        _STORED_RESULTS[f"{split}_{trip_id}"] = result

        return ReconstructResponse(**result)

    except FileNotFoundError as exc:
        logger.error("Trip data missing for %s: %s", trip_id, exc)

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip files not found: {exc}",
        ) from exc

    except ValueError as exc:
        logger.error(
            "Reconstruction validation error on trip %s: %s",
            trip_id,
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation failed: {exc}",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected error reconstructing trip %s",
            trip_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal reconstruction error: {str(exc)}",
        ) from exc


# ==================================================
# 10. RETRIEVE RECONSTRUCTION RESULTS
# ==================================================

@app.get(
    "/results/{trip_id}",
    response_model=ReconstructResponse,
    tags=["Reconstruction"],
)
def get_results(
    trip_id: str,
    split: str = "student_test",
) -> ReconstructResponse:
    """Retrieve cached results or reconstruct the trip."""

    cache_key = f"{split}_{trip_id.zfill(8)}"

    if cache_key in _STORED_RESULTS:
        return ReconstructResponse(**_STORED_RESULTS[cache_key])

    return reconstruct(
        ReconstructRequest(trip_id=trip_id, split=split)
    )


# ==================================================
# 11. PREDICTION SERVICE ENDPOINT
# ==================================================

@app.post("/predict/")
@app.post("/predict")
def predict_route(request: PredictionRequest):
    """Run the prediction service for a student-testing trip."""

    trip_id = request.trip_id

    logger.info("Prediction requested for trip %s", trip_id)

    try:
        result = prediction_service.predict_trip(trip_id)

        logger.info(
            "Prediction successfully completed for trip %s",
            trip_id,
        )

        return {
            "status": "success",
            **result,
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=501,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected prediction failure for trip %s",
            trip_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Internal prediction error",
        ) from exc


# ==================================================
# 12. MODEL ARTIFACT STATUS
# ==================================================

@app.get("/artifacts/status")
def artifact_status():
    """Check availability of required model artifacts."""

    artifacts = get_artifact_status()

    all_available = all(
        value == "available"
        for value in artifacts.values()
    )

    logger.info("Artifact availability checked: %s", artifacts)

    return {
        "artifacts_ready": all_available,
        "artifacts": artifacts,
    }


# ==================================================
# 13. PREDICTION PIPELINE STATUS
# ==================================================

@app.get("/pipeline/status")
def pipeline_status():
    """Check readiness of prediction pipeline components."""

    components = prediction_service.check_readiness()

    ready = all(components.values())

    logger.info(
        "Prediction pipeline readiness checked: %s",
        components,
    )

    return {
        "pipeline_ready": ready,
        "components": components,
    }


# ==================================================
# 14. STATIC FRONTEND
# ==================================================

frontend_dir = Path("frontend")

if frontend_dir.is_dir() and (frontend_dir / "index.html").is_file():
    app.mount(
        "/",
        StaticFiles(directory=str(frontend_dir), html=True),
        name="frontend",
    )
