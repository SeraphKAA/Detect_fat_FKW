from fastapi import FastAPI, HTTPException, UploadFile, File

from app.schema.dto import PredictionResultList
from app.model.fat_detection import FAT_DETECT


app = FastAPI(
    title="Model FAT Prediction API",
    description="API для предсказания процента жира на основе фотографии.",
    version="1.0.1",
)


@app.post(
    "/predict/fat-detection",
    summary="Предсказание плана",
    description="Предсказание процента жира",
    response_model=PredictionResultList,
)
async def detect_fat(image: UploadFile):
    allowed_types = {"image/png", "image/jpeg"}

    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=404, detail="Разрешены только изображения формата PNG или JPG"
        )

    return await FAT_DETECT(image)
