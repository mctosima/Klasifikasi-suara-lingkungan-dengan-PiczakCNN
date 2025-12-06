import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

# Impor kelas-kelas dari file Anda
from datareader import AudioDataset
from piczakCNN import PiczakCNN
from utils import check_set_gpu, collate_fn

class AudioAblationRunner:
    def __init__(self, project_name="piczak-cnn-ablation"):
        self.project_name = project_name
        self.device = check_set_gpu()
        
        # Konfigurasi terbaik (baseline) Anda dari train.py
        self.best_hp = dict(
            optimizer="sgd",
            learning_rate=0.002,
            momentum=0.9,
            epochs=25,
            batch_size=32,
            num_folds=5,
            dropout_rate=0.5,
            apply_augment=True
        )

    def _run_one_fold(self, fold, cfg, variant_name):
        """Menjalankan training dan validasi untuk satu fold dari satu eksperimen."""
        print(f"--- Running Variant: {variant_name} | Fold: {fold+1}/{cfg['num_folds']} ---")

        # 1. Load Data
        data_dir = os.path.join(os.getcwd(), 'data')
        train_ds = AudioDataset(root_dir=data_dir, fold=fold, split_type='train', apply_augment=cfg['apply_augment'])
        val_ds = AudioDataset(root_dir=data_dir, fold=fold, split_type='val', apply_augment=False) # Augmentasi selalu mati untuk validasi
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_fn)

        # 2. Build Model
        model = PiczakCNN(num_classes=2, dropout_rate=cfg['dropout_rate']).to(self.device)

        # 3. Create Optimizer
        opt_name = cfg['optimizer'].lower()
        if opt_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=cfg['learning_rate'], momentum=cfg['momentum'], nesterov=True)
        elif opt_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        else:
            raise ValueError(f"Optimizer {opt_name} tidak dikenali.")
        
        criterion = nn.CrossEntropyLoss()

        # 4. Init W&B
        run = wandb.init(
            project=self.project_name,
            name=f"{variant_name}-fold{fold+1}",
            group=f"ablation-{variant_name}",
            config=cfg,
            reinit=True
        )

        # 5. Training Loop (disederhanakan dari train.py)
        best_val_acc = 0.0
        for epoch in range(cfg['epochs']):
            model.train()
            for specs, labels, _ in train_loader:
                specs, labels = specs.to(self.device), labels.to(self.device).long()
                optimizer.zero_grad()
                outputs = model(specs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            all_labels, all_preds = [], []
            with torch.no_grad():
                for specs, labels, _ in val_loader:
                    specs, labels = specs.to(self.device), labels.to(self.device).long()
                    outputs = model(specs)
                    preds = torch.argmax(outputs, dim=1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
            
            val_acc = (np.array(all_labels) == np.array(all_preds)).mean()
            wandb.log({'epoch': epoch, 'val/accuracy': val_acc})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        run.summary['best_val_accuracy'] = best_val_acc
        wandb.finish()

        return best_val_acc

    def run(self):
        """Mendefinisikan dan menjalankan semua eksperimen ablasi."""  
        base = self.best_hp.copy()

        # Daftar eksperimen yang akan dijalankan
        experiments = [
            ("baseline", base.copy()),
            ("no_augmentation", {**base, "apply_augment": False}),
            ("adam_optimizer", {**base, "optimizer": "adam", "learning_rate": 0.001}),
            ("no_dropout", {**base, "dropout_rate": 0.0})
        ]

        summary = {}
        for variant, config in experiments:
            fold_scores = []
            for fold in range(config['num_folds']):
                score = self._run_one_fold(fold, config, variant)
                fold_scores.append(score)
            
            avg_acc = float(np.mean(fold_scores)) if fold_scores else 0.0
            summary[variant] = avg_acc
            print(f"[{variant}] Rata-rata Akurasi Validasi: {avg_acc:.4f}")

        print("\n=== RINGKASAN ABLASI (Rata-rata Akurasi) ===")
        for k, v in summary.items():
            print(f"- {k}: {v:.4f}")

if __name__ == "__main__":
    runner = AudioAblationRunner()
    runner.run()