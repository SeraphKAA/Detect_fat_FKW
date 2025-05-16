FROM python:3.12

WORKDIR /app/Detect_fat_FKW

COPY requirements.txt ./

RUN pip install --no-cache-dir ultralytics==8.3.135
RUN apt-get update && apt-get install -y libgl1

RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.org/simple

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5151"]