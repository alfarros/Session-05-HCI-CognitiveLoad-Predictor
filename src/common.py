import random
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def set_seed(seed=42):
    """Menyetel seed untuk semua library agar eksperimen dapat diulang persis sama."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_preprocess_data(filepath, seed=42):
    """
    Membaca data, membagi train/val/test, dan melakukan scaling.
    MENCEGAH DATA LEAKAGE: Standard Scaler HANYA di-fit pada data Train!
    """
    df = pd.read_csv(filepath)
    
    X = df.drop(columns=['high_cognitive_load']).values
    y = df['high_cognitive_load'].values

    # Pembagian Data Pertama: Train (70%) dan Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )

    # Pembagian Data Kedua: Temp dibagi 50/50 menjadi Val (15%) dan Test (15%)
    # Menggunakan y_temp agar konsisten dengan ukuran sampel X_temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    # MENCEGAH LEAKAGE: Scaler HANYA mempelajari rata-rata & std dari X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)      # Menggunakan fit dari Train
    X_test_scaled = scaler.transform(X_test)    # Menggunakan fit dari Train

    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test), scaler