FROM python:3.10-slim

# Tentukan folder kerja di dalam container
WORKDIR /app

# Salin berkas kebutuhan library
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode program dan model lokal kamu ke dalam container
COPY . .

# Buka jalur port 5001 untuk akses API model
EXPOSE 5001

# Jalankan perintah serving MLflow saat container dinyalakan
CMD ["mlflow", "models", "serve", "-m", "temp_outputs/model", "--host", "0.0.0.0", "--port", "5001", "--env-manager", "local"]