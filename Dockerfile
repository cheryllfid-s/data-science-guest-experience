FROM python:3.10-slim 

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY /data /data
COPY /models /models
CMD ["python3", "src/main.py"]
