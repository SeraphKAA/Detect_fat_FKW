from pydantic import BaseModel
from typing import List, Optional

class PredictionItem(BaseModel):
    predicted_class: int | None
    confidence: float | None


class PredictionResultList(BaseModel):
    predictions: List[PredictionItem]