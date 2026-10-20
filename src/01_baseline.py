import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from common import set_seed, load_and_preprocess_data

SEEDS = [42, 123, 999]
DATA_PATH = os.path.join('dataset', 'ml_quality_synthetic.csv')

def run_baseline_experiments():
    print("=== MENJALANKAN EXPERIMEN BASELINE (Isolation Forest) ===")
    
    metrics_summary = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []}
    last_cm = None

    for seed in SEEDS:
        set_seed(seed)
        (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = load_and_preprocess_data(DATA_PATH, seed=seed)

        # Isolation Forest: Algoritma unsupervised standar untuk deteksi anomali
        # contamination=0.15 disesuaikan dengan proporsi anomali asumsi
        model = IsolationForest(contamination=0.15, random_state=seed)
        model.fit(X_train)

        # Predict: IsolationForest mengembalikan 1 untuk normal, -1 untuk anomali.
        # Konversi ke standar kita: 1 = Anomali, 0 = Normal
        raw_preds = model.predict(X_test)
        y_pred = np.where(raw_preds == -1, 1, 0)
        
        # Anomaly Score (makin negatif score_samples, makin anomali)
        y_prob = -model.score_samples(X_test)

        metrics_summary['acc'].append(accuracy_score(y_test, y_pred))
        metrics_summary['prec'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics_summary['rec'].append(recall_score(y_test, y_pred, zero_division=0))
        metrics_summary['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        metrics_summary['auc'].append(roc_auc_score(y_test, y_prob))

        last_cm = confusion_matrix(y_test, y_pred)

    print("\n--- HASIL EVALUASI BASELINE (3 SEED) ---")
    for metric, values in metrics_summary.items():
        print(f"{metric.upper():<5}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(last_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomali'], yticklabels=['Normal', 'Anomali'])
    plt.title('Baseline (Isolation Forest) Confusion Matrix')
    plt.ylabel('Kenyataan (True)')
    plt.xlabel('Prediksi (Predicted)')
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'baseline_cm.png'))
    plt.close()
    print("Grafik confusion matrix disimpan ke results/baseline_cm.png\n")

if __name__ == '__main__':
    run_baseline_experiments()