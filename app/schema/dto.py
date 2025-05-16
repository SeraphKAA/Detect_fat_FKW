from pydantic import BaseModel
from typing import List, Optional

class PredictionItem(BaseModel):
    predicted_class: int | None
    confidence: float | None


class PredictionResultList(BaseModel):
    image_path: Optional[str]
    predictions: List[PredictionItem]