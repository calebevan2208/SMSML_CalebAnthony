"""
Eksperimen_CalebAnthony.py
--------------------------
Modul ini menjalankan Exploratory Data Analysis (EDA) tingkat lanjut (Advanced).
Menghasilkan artifacts visual untuk analisis perilaku nasabah (Behavioral Analysis)
dan identifikasi pola Churn (Pattern Recognition).

Author: Caleb Anthony (Automated by System)
Date: 2025-10-30
Version: 2.1 (Comprehensive Visualization Edition)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import logging
from pathlib import Path
from typing import Optional, List

# --- KONFIGURASI LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- KONFIGURASI PROJECT ---
class AnalysisConfig:
    """
    Konfigurasi sentral untuk parameter analisis dan visualisasi.
    """
    BASE_DIR = Path(__file__).resolve().parent
    INPUT_DATA_PATH = BASE_DIR / 'churn_preprocessing' / 'clean_data.csv'
    OUTPUT_DIR = BASE_DIR / 'analysis_results'
    
    # Palette warna profesional: 'coolwarm' untuk heatmap, 'viridis' atau custom untuk kategori
    COLOR_PALETTE = {0: '#2ecc71', 1: '#e74c3c'} # Hijau (Aman), Merah (Churn)
    PLOT_STYLE = 'whitegrid'

class ChurnAdvancedEDA:
    """
    Kelas Orchestrator untuk EDA mendalam.
    Menggunakan pendekatan OOP untuk memisahkan logika loading, processing, dan plotting.
    """

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        
        # Setup direktori & Style
        AnalysisConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style=AnalysisConfig.PLOT_STYLE)
        
        # Mapping label untuk plot agar lebih mudah dibaca manajemen
        self.label_map = {0: 'Non-Churn', 1: 'Churn'}
        
        logger.info(f"Advanced EDA initialized. Output path: {AnalysisConfig.OUTPUT_DIR}")

    def load_data(self) -> None:
        """Memuat data bersih."""
        if not AnalysisConfig.INPUT_DATA_PATH.exists():
            logger.critical(f"Dataset tidak ditemukan di {AnalysisConfig.INPUT_DATA_PATH}")
            sys.exit(1)
        
        try:
            self.df = pd.read_csv(AnalysisConfig.INPUT_DATA_PATH)
            # Buat kolom helper 'Label' untuk legend plot yang lebih cantik
            self.df['Label'] = self.df['default'].map(self.label_map)
            logger.info(f"Data dimuat: {self.df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def save_descriptive_stats(self) -> None:
        """Menyimpan statistik dasar ke text file."""
        output_path = AnalysisConfig.OUTPUT_DIR / '0_descriptive_stats.txt'
        with open(output_path, 'w') as f:
            f.write("=== DATASET OVERVIEW (INFO) ===\n")
            self.df.info(buf=f)
            
            f.write("\n\n=== MISSING VALUES (ISNA) ===\n")
            f.write(self.df.isna().sum().to_string())
            
            f.write("\n\n=== NUMERICAL SUMMARY (DESCRIBE) ===\n")
            f.write(self.df.describe().to_string())

            f.write("\n\n=== VALUE COUNTS (CATEGORICAL) ===\n")
            # Kolom kategorikal yang relevan
            cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'Label', 'default']
            for col in cat_cols:
                if col in self.df.columns:
                    f.write(f"\n--- {col} ---\n")
                    f.write(self.df[col].value_counts().to_string())
            
            f.write("\n\n=== CORRELATION MATRIX (NUMERICAL) ===\n")
            numeric_df = self.df.select_dtypes(include=['number'])
            f.write(numeric_df.corr().to_string())

        logger.info("Statistik deskriptif tersimpan.")

    # --- PLOT 1: TARGET DISTRIBUTION (STANDARD) ---
    def plot_target_distribution(self):
        plt.figure(figsize=(7, 5))
        ax = sns.countplot(x='Label', data=self.df, palette=['#2ecc71', '#e74c3c'], order=['Non-Churn', 'Churn'])
        
        plt.title('Proporsi Nasabah Churn vs Non-Churn', fontsize=14, fontweight='bold')
        plt.xlabel('')
        plt.ylabel('Jumlah Nasabah')
        
        # Annotate percentages
        total = len(self.df)
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_height() / total)
            x = p.get_x() + p.get_width() / 2 - 0.1
            y = p.get_height() + 100
            ax.annotate(percentage, (x, y), size=12, weight='bold')
            
        plt.savefig(AnalysisConfig.OUTPUT_DIR / '1_target_distribution.png')
        plt.close()
        logger.info("[Plot 1] Target distribution saved.")

    # --- PLOT 2: CORRELATION RANKING (NEW & INSIGHTFUL) ---
    def plot_feature_correlation_ranking(self):
        """
        Membuat Bar Chart horizontal yang mengurutkan fitur berdasarkan korelasi dengan Target.
        Jauh lebih mudah dibaca daripada Heatmap raksasa.
        """
        logger.info("Menghitung korelasi fitur...")
        
        # Drop kolom non-numerik dan target itu sendiri
        numeric_df = self.df.select_dtypes(include=['number']).drop(columns=['default'])
        
        # Hitung korelasi dengan target 'default'
        correlations = numeric_df.corrwith(self.df['default']).sort_values(ascending=True)
        
        # Plot
        plt.figure(figsize=(10, 8))
        colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in correlations.values] # Merah jika korelasi positif (risiko), Hijau jika negatif
        
        correlations.plot(kind='barh', color=colors)
        plt.title('Faktor Pendorong Churn (Correlation Ranking)', fontsize=15, fontweight='bold')
        plt.xlabel('Koefisien Korelasi (Pearson)')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        plt.savefig(AnalysisConfig.OUTPUT_DIR / '2_feature_importance_corr.png')
        plt.close()
        logger.info("[Plot 2] Feature correlation ranking saved.")

    # --- PLOT 3: PAYMENT BEHAVIOR TREND (NEW & ADVANCED) ---
    def plot_payment_trend(self):
        """
        Menganalisis tren perilaku pembayaran dari 6 bulan lalu (PAY_6) sampai bulan ini (PAY_1).
        Ini menjawab: "Apakah kondisi keuangan mereka memburuk seiring waktu?"
        """
        # Kolom pembayaran diurutkan secara kronologis: April (PAY_6) -> September (PAY_1)
        pay_cols = ['PAY_6', 'PAY_5', 'PAY_4', 'PAY_3', 'PAY_2', 'PAY_1']
        
        # Cek ketersediaan kolom
        if not all(col in self.df.columns for col in pay_cols):
            logger.warning("Kolom PAY_1 - PAY_6 tidak lengkap. Melewati analisis tren.")
            return

        # Menghitung rata-rata status pembayaran per grup
        trend_data = self.df.groupby('Label')[pay_cols].mean().T
        
        plt.figure(figsize=(10, 6))
        
        # Plot Non-Churn
        plt.plot(trend_data.index, trend_data['Non-Churn'], marker='o', label='Non-Churn', 
                 color='#2ecc71', linewidth=2, markersize=8)
        
        # Plot Churn
        plt.plot(trend_data.index, trend_data['Churn'], marker='o', label='Churn', 
                 color='#e74c3c', linewidth=2, markersize=8)
        
        plt.title('Tren Keterlambatan Pembayaran (6 Bulan Terakhir)', fontsize=15, fontweight='bold')
        plt.ylabel('Rata-rata Status Pembayaran\n(Semakin tinggi = Semakin telat)', fontsize=12)
        plt.xlabel('Bulan (PAY_6 = April, PAY_1 = September)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(AnalysisConfig.OUTPUT_DIR / '3_payment_trend_analysis.png')
        plt.close()
        logger.info("[Plot 3] Payment trend analysis saved.")

    # --- PLOT 4: VIOLIN PLOT LIMIT BALANCE (NEW) ---
    def plot_limit_balance_violin(self):
        """
        Violin plot menggabungkan Boxplot dan KDE.
        Sangat bagus untuk melihat densitas nasabah di berbagai level Limit Balance.
        """
        plt.figure(figsize=(10, 6))
        
        sns.violinplot(data=self.df, x='Label', y='LIMIT_BAL', palette={'Non-Churn': '#2ecc71', 'Churn': '#e74c3c'},
                       order=['Non-Churn', 'Churn'], split=False)
        
        plt.title('Distribusi Limit Kartu Kredit (Violin Plot)', fontsize=15, fontweight='bold')
        plt.ylabel('Limit Balance (NT Dollar)')
        plt.xlabel('Status')
        
        # Format y-axis agar tidak notasi ilmiah (e.g., 1e6)
        current_values = plt.gca().get_yticks()
        plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
        
        plt.savefig(AnalysisConfig.OUTPUT_DIR / '4_limit_balance_violin.png')
        plt.close()
        logger.info("[Plot 4] Limit balance violin plot saved.")

    # --- PLOT 5: DEMOGRAPHIC SEGMENTATION (SCATTER) ---
    def plot_age_limit_interaction(self):
        """
        Melihat interaksi antara Umur dan Limit.
        Biasanya anak muda limit kecil, orang tua limit besar. Di mana posisi Churners?
        Kita gunakan sampel acak 2000 data agar plot tidak terlalu berat (overplotting).
        """
        sample_df = self.df.sample(n=min(2000, len(self.df)), random_state=42)
        
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=sample_df, x='AGE', y='LIMIT_BAL', hue='Label', 
                        palette={'Non-Churn': '#2ecc71', 'Churn': '#e74c3c'}, alpha=0.6)
        
        plt.title('Segmentasi Demografis: Umur vs Limit (Sample 2000)', fontsize=15, fontweight='bold')
        plt.xlabel('Umur (Tahun)')
        plt.ylabel('Limit Balance')
        
        plt.savefig(AnalysisConfig.OUTPUT_DIR / '5_demographic_scatter.png')
        plt.close()
        logger.info("[Plot 5] Demographic scatter plot saved.")

    # --- PLOT 6: NUMERICAL DISTRIBUTIONS (HISTOGRAM) ---
    def plot_numerical_distributions(self):
        """
        Distribusi fitur numerik utama (AGE, LIMIT_BAL).
        """
        cols = ['AGE', 'LIMIT_BAL']
        plt.figure(figsize=(14, 6))
        
        for i, col in enumerate(cols, 1):
            plt.subplot(1, 2, i)
            sns.histplot(data=self.df, x=col, hue='Label', kde=True, 
                         palette={'Non-Churn': '#2ecc71', 'Churn': '#e74c3c'}, multiple="stack")
            plt.title(f'Distribusi {col}', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(AnalysisConfig.OUTPUT_DIR / '6_numerical_distributions.png')
        plt.close()
        logger.info("[Plot 6] Numerical distributions saved.")

    # --- PLOT 7: CATEGORICAL DISTRIBUTIONS (COUNT PLOT) ---
    def plot_categorical_distributions(self):
        """
        Distribusi fitur kategorikal (SEX, EDUCATION, MARRIAGE).
        """
        cols = ['SEX', 'EDUCATION', 'MARRIAGE']
        if not all(col in self.df.columns for col in cols):
             logger.warning("Kolom kategorikal tidak lengkap. Melewati analisis kategorikal.")
             return

        plt.figure(figsize=(18, 6))
        for i, col in enumerate(cols, 1):
            plt.subplot(1, 3, i)
            sns.countplot(x=col, data=self.df, hue='Label', palette={'Non-Churn': '#2ecc71', 'Churn': '#e74c3c'})
            plt.title(f'Distribusi {col} per Status Churn', fontsize=12, fontweight='bold')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.legend(title='Status')
            
        plt.tight_layout()
        plt.savefig(AnalysisConfig.OUTPUT_DIR / '7_categorical_distributions.png')
        plt.close()
        logger.info("[Plot 7] Categorical distributions saved.")

    # --- PLOT 8: CORRELATION HEATMAP ---
    def plot_correlation_heatmap(self):
        """
        Heatmap korelasi lengkap antar semua fitur numerik.
        """
        numeric_df = self.df.select_dtypes(include=['number'])
        plt.figure(figsize=(16, 12))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix Heatmap', fontsize=15, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(AnalysisConfig.OUTPUT_DIR / '8_correlation_heatmap.png')
        plt.close()
        logger.info("[Plot 8] Correlation heatmap saved.")

    # --- PLOT 9: OUTLIER ANALYSIS (BOX PLOT) ---
    def plot_outlier_analysis(self):
        """
        Analisis outlier untuk fitur pembayaran dan tagihan.
        """
        bill_cols = [c for c in self.df.columns if 'BILL_AMT' in c]
        pay_cols = [c for c in self.df.columns if 'PAY_AMT' in c]
        
        if bill_cols:
            plt.figure(figsize=(14, 6))
            sns.boxplot(data=self.df[bill_cols], orient='h', palette='Set2')
            plt.title('Distribusi Outlier: Bill Amount', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(AnalysisConfig.OUTPUT_DIR / '9_outlier_bill.png')
            plt.close()
            
        if pay_cols:
            plt.figure(figsize=(14, 6))
            sns.boxplot(data=self.df[pay_cols], orient='h', palette='Set3')
            plt.title('Distribusi Outlier: Payment Amount', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(AnalysisConfig.OUTPUT_DIR / '9_outlier_pay.png')
            plt.close()
            
        logger.info("[Plot 9] Outlier analysis saved.")


    def run(self):
        """Eksekusi pipeline analisis."""
        logger.info("=== STARTING ADVANCED EDA ===")
        
        self.load_data()
        self.save_descriptive_stats()
        
        # Visualizations
        self.plot_target_distribution()         # Basic
        self.plot_feature_correlation_ranking() # Insightful
        self.plot_payment_trend()               # Time-Series Behavior
        self.plot_limit_balance_violin()        # Distribution
        self.plot_age_limit_interaction()       # Multivariate
        
        # New Detailed Visualizations
        self.plot_numerical_distributions()
        self.plot_categorical_distributions()
        self.plot_correlation_heatmap()
        self.plot_outlier_analysis()
        
        logger.info("=== EDA COMPLETED ===")
        logger.info(f"Semua plot telah disimpan di folder: {AnalysisConfig.OUTPUT_DIR}")

if __name__ == "__main__":
    eda = ChurnAdvancedEDA()
    eda.run()
