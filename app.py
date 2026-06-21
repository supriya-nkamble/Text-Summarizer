import subprocess
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from starlette.responses import RedirectResponse

from text_summarizer.pipeline.prediction import PredictionPipeline

app = FastAPI(title="Text Summarizer", description="Fine-tuned Pegasus summarization API")

_MIN_TEXT_LENGTH = 50
_MAX_TEXT_LENGTH = 10_000


@app.get("/", tags=["health"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train", tags=["training"])
async def training():
    result = subprocess.run(
        [sys.executable, "main.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr[-2000:] or "Training failed")
    return Response("Training successful")


@app.post("/predict", tags=["inference"])
async def predict_route(text: str):
    if len(text) < _MIN_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Input too short — provide at least {_MIN_TEXT_LENGTH} characters",
        )
    if len(text) > _MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Input too long — maximum {_MAX_TEXT_LENGTH} characters",
        )
    pipeline = PredictionPipeline()
    return pipeline.predict(text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
