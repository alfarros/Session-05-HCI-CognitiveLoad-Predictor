import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from common import set_seed, load_and_preprocess_data

SEEDS = [42, 123, 999]
DATA_PATH = os.path.join('dataset', 'hci_cognitive_load_synthetic.csv')
DEVICE = torch.device('cpu')

# Arsitektur PyTorch MLP untuk Memprediksi Beban Kognitif Desain
class CognitiveLoadNet(nn.Module):
    def __init__(self, input_dim):
        super(CognitiveLoadNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1) # Output Logit
        )

    def forward(self, x):
        return self.net(x)

def run_dl_experiments():
    print("=== MENJALANKAN EXPERIMEN DEEP LEARNING (PyTorch MLP) ===")
    
    metrics_summary = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': []}
    last_cm = None
    last_history = None

    for seed in SEEDS:
        set_seed(seed)
        (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = load_and_preprocess_data(DATA_PATH, seed=seed)

        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(1))
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        model = CognitiveLoadNet(input_dim=X_train.shape[1]).to(DEVICE)

        # Penanganan Imbalanced Data menggunakan pos_weight
        num_pos = np.sum(y_train == 1)
        num_neg = np.sum(y_train == 0)
        pos_weight = torch.tensor([num_neg / (num_pos + 1e-5)]).to(DEVICE)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_weights = None

        history = {'train_loss': [], 'val_loss': []}

        # Training Loop
        for epoch in range(100):
            model.train()
            train_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                out = model(X_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_b.size(0)
            
            train_loss /= len(train_loader.dataset)

            # Validation Loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                    out = model(X_b)
                    loss = criterion(out, y_b)
                    val_loss += loss.item() * X_b.size(0)
            
            val_loss /= len(val_loader.dataset)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # Early Stopping Check berdasarkan Validation Loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_weights = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(best_model_weights)
        last_history = history

        # --- THRESHOLD TUNING PADA VALIDATION SET ---
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.FloatTensor(X_val).to(DEVICE))
            val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()

        best_thresh = 0.5
        best_val_f1 = 0.0
        for thresh in np.arange(0.1, 0.9, 0.05):
            score = f1_score(y_val, (val_probs >= thresh).astype(int), zero_division=0)
            if score > best_val_f1:
                best_val_f1 = score
                best_thresh = thresh

        # --- EVALUASI AKHIR PADA TEST SET ---
        with torch.no_grad():
            test_logits = model(torch.FloatTensor(X_test).to(DEVICE))
            test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
            test_preds = (test_probs >= best_thresh).astype(int)

        metrics_summary['acc'].append(accuracy_score(y_test, test_preds))
        metrics_summary['prec'].append(precision_score(y_test, test_preds, zero_division=0))
        metrics_summary['rec'].append(recall_score(y_test, test_preds, zero_division=0))
        metrics_summary['f1'].append(f1_score(y_test, test_preds, zero_division=0))
        metrics_summary['auc'].append(roc_auc_score(y_test, test_probs))
        
        last_cm = confusion_matrix(y_test, test_preds)

    print("\n--- HASIL EVALUASI DEEP LEARNING (3 SEED) ---")
    for metric, values in metrics_summary.items():
        print(f"{metric.upper():<5}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(last_history['train_loss'], label='Train Loss')
    plt.plot(last_history['val_loss'], label='Val Loss')
    plt.title('Kurva Training & Validation Loss (HCI Predictor)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'dl_loss_curve.png'))
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.heatmap(last_cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Intuitif', 'Beban Tinggi'], yticklabels=['Intuitif', 'Beban Tinggi'])
    plt.title('Deep Learning Confusion Matrix')
    plt.ylabel('Kenyataan (True)')
    plt.xlabel('Prediksi (Predicted)')
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'dl_cm.png'))
    plt.close()
    print("Grafik kurva loss dan confusion matrix disimpan di direktori results/\n")

if __name__ == '__main__':
    run_dl_experiments()