# Base image
FROM python:3.11-slim

# Nastavíme pracovný priečinok
WORKDIR /app

# Skopírujeme requirements
COPY requirements.txt .

# Upgrade pip a inštalácia dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Skopírujeme všetky zdrojové súbory
COPY *.py ./

CMD ["python", "extractAndRotatePhotos.py"]
