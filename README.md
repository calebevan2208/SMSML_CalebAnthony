<<<<<<< HEAD
# SMSML CalebAnthony Project

Project ini adalah implementasi end-to-end Machine Learning pipeline untuk prediksi Churn Nasabah kartu kredit. Project ini mencakup tahapan Ingestion data, Preprocessing otomatis, Exploratory Data Analysis (EDA) yang komprehensif, Modeling dengan Deep Learning (TensorFlow), dan integrasi CI/CD dengan GitHub Actions serta MLflow.

## 📂 Struktur Project

```
SMSML_CalebAnthony/
│
├── Eksperimen_SML_CalebAnthony/      # [Tahap 1] Eksperimen & Preprocessing
│   ├── churn_raw/                    # Folder penyimpanan raw data (diunduh otomatis)
│   ├── churn_preprocessing/          # Folder output data bersih
│   ├── eksperimen_CalebAnthony.py    # Script EDA (Visualisasi & Analisis)
│   ├── automate_CalebAnthony.py      # Script Automation (Download & Cleaning)
│   └── requirements.txt              # Dependencies khusus eksperimen
│
├── Membangun_model/                  # [Tahap 2] Training & Tracking
│   ├── churn_preprocessing/          # Salinan data bersih untuk training
│   ├── modelling.py                  # Script Baseline Model Training (TensorFlow)
│   ├── modelling_tuning.py           # Script Hyperparameter Tuning
│   └── requirements.txt              # Dependencies untuk modeling & MLflow
│
└── Workflow-CI/                      # [Tahap 3] CI/CD & MLflow Configuration
    ├── .github/workflows/            # (Note: Workflow file moved to root .github/workflows for functionality)
    └── MLProject/
        ├── MLproject                 # Konfigurasi Entry Point MLflow
        ├── conda.yaml                # Konfigurasi Environment MLflow
        └── modelling.py              # Script Modeling yang diadaptasi untuk MLflow
```

## 🚀 Cara Menjalankan Project

### 1. Persiapan Environment
Pastikan Anda menggunakan Python 3.10 atau lebih baru.
```bash
pip install -r SMSML_CalebAnthony/Membangun_model/requirements.txt
```

### 2. Data Automation (Ingestion & Cleaning)
Jalankan script ini untuk mengunduh data dari sumber dan membersihkannya. Script ini juga akan menyalin data bersih ke folder `Membangun_model`.
```bash
python SMSML_CalebAnthony/Eksperimen_SML_CalebAnthony/automate_CalebAnthony.py
```
*Output: `clean_data.csv` di folder `Eksperimen.../churn_preprocessing` dan `Membangun_model/churn_preprocessing`.*

### 3. Exploratory Data Analysis (EDA)
Jalankan script ini untuk menghasilkan visualisasi lengkap tentang distribusi data, korelasi, dan pola churn.
```bash
python SMSML_CalebAnthony/Eksperimen_SML_CalebAnthony/eksperimen_CalebAnthony.py
```
*Output: File gambar (.png) dan statistik (.txt) di folder `analysis_results`.*

### 4. Model Training (Baseline)
Jalankan training model baseline Deep Learning.
```bash
python SMSML_CalebAnthony/Membangun_model/modelling.py
```
*Output: Model (`baseline_model.h5`), Scaler (`scaler.pkl`), dan plot history training di folder `artifacts`.*

### 5. MLflow Run (CLI)
Untuk menjalankan project menggunakan MLflow dengan parameter kustom:
```bash
cd SMSML_CalebAnthony/Workflow-CI/MLProject
python modelling.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

## 🔄 CI/CD Pipeline
Project ini dilengkapi dengan GitHub Actions workflow (`.github/workflows/main.yml`) yang berjalan otomatis pada setiap Push atau Pull Request ke branch `main`.

**Pipeline Steps:**
1.  **Setup Environment**: Install Python 3.10 dan dependencies.
2.  **Data Pipeline**: Menjalankan `automate_CalebAnthony.py` untuk memastikan data source dapat diakses dan diproses dengan benar.
3.  **Model Training**: Menjalankan training model via script MLflow untuk memastikan kode modeling bebas error (Smoke Test).

## 🛠 Teknologi Utama
-   **Python 3.10**
-   **TensorFlow/Keras**: Deep Learning Framework
-   **Pandas & NumPy**: Data Manipulation
-   **Matplotlib & Seaborn**: Data Visualization
-   **MLflow**: Experiment Tracking
-   **GitHub Actions**: CI Automation
=======
# SMSML_CalebAnthony
>>>>>>> 3ab07050e90340e24e8e20c4f8ba67667770197c
