from ultralytics import YOLO
from fastapi import UploadFile
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import numpy as np


from app.schema.dto import PredictionResult

MODEL = YOLO("app/models_pt/best.pt")


async def FAT_DETECT(image: UploadFile) -> PredictionResult:
    # Читаем файл в память
    contents = await image.read()
    img_stream = BytesIO(contents)

    # Преобразуем в массив numpy (YOLO принимает np.ndarray, PIL.Image, путь или torch.Tensor)
    pil_image = Image.open(img_stream).convert("RGB")
    np_image = np.array(pil_image)

    # Запускаем инференс
    results = MODEL(np_image, save=False)

    best_pred = None
    max_conf = 0

    for r in results:
        for cls, box, conf in zip(r.boxes.cls, r.boxes.xyxy, r.boxes.conf):
            if conf > max_conf:
                max_conf = float(conf)
                best_pred = int(cls)

    # Если ничего не найдено
    if best_pred is None:
        return PredictionResult(
            image_path=str("sometihng result"), predicted_class=None, confidence=None
        )

    return PredictionResult(
        image_path=str("sometihng result"),
        predicted_class=best_pred,
        confidence=max_conf,
    )
