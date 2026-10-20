import os
import numpy as np
import pandas as pd

# Menyetel seed agar dataset sintetik dapat diulang dengan hasil persis sama
np.random.seed(42)

def generate_cognitive_load_data(n_samples=2000):
    """
    Membuat dataset sintetik metrik elemen visual antarmuka/dasbor.
    Target (y): 1 jika Beban Kognitif Tinggi / Extraneous Load Buruk (Desain Membingungkan), 
                0 jika Beban Kognitif Rendah / Intuitif.
    """
    print(f"Membuat {n_samples} sampel data sintetik metrik antarmuka IMK...")
    
    # Fitur-fitur Metrik Desain Antarmuka
    jumlah_grafik = np.random.randint(1, 12, size=n_samples)             # Jumlah elemen visual/chart di 1 layar
    rasio_kontras = np.random.uniform(1.5, 12.0, size=n_samples)          # Kontras warna teks (Standar WCAG > 4.5)
    data_ink_ratio = np.random.uniform(0.1, 0.95, size=n_samples)        # Efisiensi Tufte (makin tinggi makin bersih)
    kerapatan_tombol = np.random.uniform(10.0, 85.0, size=n_samples)     # Kepadatan tombol/navigasi per area (%)
    hirarki_teks = np.random.randint(1, 6, size=n_samples)               # Kejelasan hirarki visual (skala 1-5)

    # Logika Psikologi Kognitif IMK (Fungsi Non-Linear dengan Noise)
    # Beban Kognitif Tinggi jika: Banyak chartjunk + Kontras buruk + Data-Ink Ratio rendah + Layar terlalu padat
    skor_beban = (
        (jumlah_grafik / 12.0) * 3.0 +
        (np.where(rasio_kontras < 4.5, 2.5, 0.2)) +
        ((1.0 - data_ink_ratio) * 2.5) +
        (kerapatan_tombol / 85.0) * 2.0 -
        (hirarki_teks / 5.0) * 1.5
    )
    
    # Menambahkan noise variasi persepsi individu pengguna
    noise = np.random.normal(0, 0.5, size=n_samples)
    final_score = skor_beban + noise
    
    # Tentukan threshold agar sekitar 35% desain tergolong Beban Kognitif Tinggi (imbalanced)
    threshold = np.percentile(final_score, 65)
    high_cognitive_load = (final_score >= threshold).astype(int)

    df = pd.DataFrame({
        'jumlah_grafik': jumlah_grafik,
        'rasio_kontras': rasio_kontras,
        'data_ink_ratio': data_ink_ratio,
        'kerapatan_tombol': kerapatan_tombol,
        'hirarki_teks': hirarki_teks,
        'high_cognitive_load': high_cognitive_load
    })

    os.makedirs('dataset', exist_ok=True)
    filepath = os.path.join('dataset', 'hci_cognitive_load_synthetic.csv')
    df.to_csv(filepath, index=False)
    print(f"Dataset metrik antarmuka berhasil disimpan di: {filepath}")
    print(f"Distribusi Beban Kognitif:\n{df['high_cognitive_load'].value_counts(normalize=True)}\n")

if __name__ == '__main__':
    generate_cognitive_load_data()