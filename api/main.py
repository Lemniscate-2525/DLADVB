from fastapi import FastAPI

from api.schemas import (
    LogRequest,
    BatchLogRequest
)

from inference.predictor import LogPredictor


app = FastAPI(
    title="Distributed Log Anomaly Detection API",
    version="1.0.0"
)

predictor = LogPredictor()


@app.get("/")
def home():

    return {
        "message": "Distributed Log Anomaly Detection API"
    }


@app.post("/predict")
def predict(request: LogRequest):

    return predictor.predict(
        request.log
    )


@app.post("/batch_predict")
def batch_predict(request: BatchLogRequest):

    return predictor.predict_batch(
        request.logs
    )