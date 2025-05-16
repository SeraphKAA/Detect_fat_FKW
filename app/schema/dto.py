from pydantic import BaseModel

class PredictionResult(BaseModel):
    image_path: str | None
    predicted_class: int | None
    confidence: float | None