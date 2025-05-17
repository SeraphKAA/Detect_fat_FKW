from fastapi import UploadFile
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import numpy as np
from pathlib import Path

from app.schema.dto import PredictionItem, PredictionResultList

MODEL = YOLO(str(Path(__file__).resolve().parent.parent / "models_pt" / "best.pt"))


async def FAT_DETECT(image: UploadFile) -> PredictionResultList:
    contents = await image.read()
    img_stream = BytesIO(contents)

    pil_image = Image.open(img_stream).convert("RGB")
    np_image = np.array(pil_image)

    results = MODEL(np_image, save=False)

    predictions = []

    for r in results:
        for cls, box, conf in zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf):
            # print("class:", cls, "; conf =", conf)
            predictions.append(
                PredictionItem(predicted_class=int(cls), confidence=float(conf))
            )

    # print(predictions)
    return PredictionResultList(predictions=predictions)
