# Session 05: HCI Cognitive Load & Usability Predictor with Deep Learning

Sistem penganalisis otomatis risiko kelelahan mental (*Extraneous Cognitive Load*) pengguna berdasarkan metrik elemen antarmuka visual menggunakan arsitektur PyTorch.

## 1. Dari Materi Pemateri vs Pengembangan Mandiri
* **Materi Pemateri (Arafat Febriandirza - BRIN):** Membahas psikologi kognitif pengguna, *Working Memory* (7±2 chunks), tiga jenis *Cognitive Load*, prinsip Gestalt, dan eliminasi *Chartjunk* (Tufte).
* **Pengembangan Mandiri:** Membangun model *Deep Learning* (PyTorch MLP) untuk menguji secara kuantitatif apakah gabungan metrik desain visual antarmuka memicu kelelahan kognitif tinggi pada pengguna.

## 2. Masalah dan Tujuan
Menilai kelelahan kognitif antarmuka secara manual melalui *Usability Testing* memakan waktu dan biaya mahal. Penelitian ini bertujuan mengomputasi metrik visual antarmuka (rasio kontras, jumlah grafik, data-ink ratio) untuk memprediksi tingkat beban mental secara otomatis sebelum produk dirilis.

## 3. Spesifikasi Dataset
* **Sumber:** Dataset Sintetik (`dataset/hci_cognitive_load_synthetic.csv`) yang disintesis melalui skrip `dataset/generate_dataset.py`.
* **Ukuran:** 2.000 sampel desain antarmuka.
* **Fitur Input:** `jumlah_grafik`, `rasio_kontras`, `data_ink_ratio`, `kerapatan_tombol`, `hirarki_teks`.
* **Target:** `high_cognitive_load` (Binary: `0` = Intuitif/Beban Rendah, `1` = Beban Kognitif Tinggi/Desain Buruk).

## 4. Metodologi & Pencegahan Leakage
1. **Pencegahan Data Leakage:** Split Train/Val/Test (70/15/15) dilakukan terlebih dahulu. `StandardScaler` di-fit **hanya** pada data Train.
2. **Penanganan Imbalanced Data:** Menggunakan `pos_weight` pada `BCEWithLogitsLoss` dan penyesuaian threshold keputusan optimal pada data Validation.
3. **Early Stopping:** Dikontrol berdasarkan pergerakan Validation Loss (patience=10) untuk mencegah *overfitting*.
4. **Reproducibility:** Eksperimen dijalankan pada 3 Seed acak (`42`, `123`, `999`).

## 5. Arsitektur / Alur Sistem
[Metrik Visual UI (Kontras, Chartjunk, Kepadatan)]-->[StandardScaler (Fit Train)]-->[PyTorch MLP: 5 -> 32 (BatchNorm+ReLU+Dropout) -> 16 -> 1]-->[Threshold Tuning (Validation Set)]-->[Threshold Tuning (Validation Set)]-->[Prediksi Usability: Intuitif / Beban Kognitif Tinggi]

## 6. Cara Menjalankan Perintah

**Di Windows (PowerShell / Command Prompt):**
```powershell
# 1. Buat dan aktifkan virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset
python dataset/generate_dataset.py

# 4. Jalankan eksperimen Baseline (Random Forest)
python src/01_baseline.py

# 5. Jalankan eksperimen Deep Learning (PyTorch)
python src/02_train_dl.py