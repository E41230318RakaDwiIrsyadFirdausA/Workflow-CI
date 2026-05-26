import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- KONFIGURASI INTEGRASI ONLINE DAGSHUB MLFLOW ---
os.environ["MLFLOW_TRACKING_USERNAME"] = "irsyaddwi30" 
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c1e0492ff2e5c2ced2152ba4cbbc0fe769f0fc69"

# URL Tracking MLflow DagsHub
mlflow.set_tracking_uri("https://dagshub.com/irsyaddwi30/MLProject_Raka.mlflow/")

def run_base_modelling():
    # 1. Membaca Dataset
    data_path = "namadataset_preprocessing/credit_score_clean.csv"
    if not os.path.exists(data_path):
        data_path = "namadataset_preprocessing"
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Mengatur Nama Eksperimen Besar di Dashboard MLflow
    mlflow.set_experiment("Eksperimen_SML_Raka-Dwi")
    
    # =========================================================================
    # [PERBAIKAN KRITERIA 2] Mengaktifkan MLflow Autolog Sebelum Training Dimulai
    # =========================================================================
    mlflow.autolog(log_models=True)
    
    
    
    with mlflow.start_run(run_name="Base_Model_RandomForest_Fixed", nested=True):
        n_estimators = 100
        
        # Training Model (Semua parameter, metrik, & folder model otomatis dicatat di sini)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi Model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # NOTE: Logging manual parameter dan metrik utama DIHAPUS atas saran reviewer
        # karena sudah di-handle sepenuhnya secara otomatis oleh mlflow.autolog()
        
        # --- KRITERIA LANJUTAN: Minimal 2 Artefak Kustom (Dipertahankan) ---
        os.makedirs("temp_outputs", exist_ok=True)
        
        # Artefak 1: Ringkasan Laporan Klasifikasi format JSON
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        with open("temp_outputs/classification_report.json", "w") as f:
            json.dump(report_dict, f, indent=4)
        mlflow.log_artifact("temp_outputs/classification_report.json")
        
        # Artefak 2: Plot Grafik Gambar Heatmap Confusion Matrix (.png)
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Raka Dwi Irsyad')
        plt.xlabel('Prediksi')
        plt.ylabel('Aktual')
        plt.tight_layout()
        plt.savefig("temp_outputs/confusion_matrix_base.png")
        plt.close()
        mlflow.log_artifact("temp_outputs/confusion_matrix_base.png")
        
        # 1. Buat folder model lokal di komputer
        os.makedirs("temp_outputs/model", exist_ok=True)
        
        # 2. Simpan model asli ke dalam folder tersebut secara lokal menggunakan MLflow
        mlflow.sklearn.save_model(model, "temp_outputs/model", serialization_format="cloudpickle")
        
        # 3. Paksa upload seluruh folder beserta isinya ke DagsHub
        mlflow.log_artifact("temp_outputs/model")
        # =========================================================================
        
        print(f"[SUCCESS] Folder model & Autolog berhasil dipaksa masuk ke DagsHub! Akurasi: {acc:.4f}")

if __name__ == "__main__":
    run_base_modelling()