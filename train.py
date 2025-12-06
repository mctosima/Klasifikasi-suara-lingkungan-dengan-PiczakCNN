import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datareader import AudioDataset
from sklearn.metrics import confusion_matrix, classification_report
from piczakCNN import PiczakCNN
import wandb
import matplotlib.pyplot as plt
import argparse
from torch.optim.lr_scheduler import StepLR  

# check_GPU file untuk otomatis mengganti device ke GPU jika tersedia
# collate_fn untuk menggabungkan batch data
from utils import check_set_gpu, collate_fn 

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint tidak ditemukan: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint dimuat dari: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Accuracy: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
    
    return checkpoint

def run_train(fold: int, resume_from_checkpoint=None):
    # Set device
    device = check_set_gpu()
    print(f"Using device: {device}")

    # Mengatur lokasi dataset
    data_dir = os.path.join(os.getcwd(), 'data')

    # Inisialisasi dataset untuk fold tertentu
    train_dataset = AudioDataset(root_dir=data_dir, fold=fold, split_type='train', segment_length=41)
    val_dataset = AudioDataset(root_dir=data_dir, fold=fold, split_type='val', segment_length=41)

    # Membuat Data Loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Inisialisasi model
    model = PiczakCNN(
        num_classes=2,
        input_shape=(60, 41)
    ).to(device)

    # Mendefinisikan Loss Function dan Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9, nesterov=True)

    # CHANGED: Scheduler StepLR (turunkan LR setiap N epoch)
    scheduler = StepLR(
        optimizer,
        step_size=5,     # turunkan LR setiap 5 epoch
        gamma=0.5        # turunkan LR menjadi 0.5x
    )

    # Init Weights & Biases
    class_names = ['ambulance', 'klakson_mobil']
    wandb.init(
        project=os.environ.get('WANDB_PROJECT', 'piczak-cnn Audio'),
        name=os.environ.get('WANDB_RUN_NAME', f'fold-{fold+1}_Baseline'),
        config={
            'epochs': 25,
            'batch_size': 32,
            'optimizer': 'SGD',
            'lr': 0.002,
            'momentum': 0.9,
            'nesterov': True,
            'model': 'PiczakCNN',
            'num_classes': 2,
            'input_shape': (60, 41),
            'segment_length': 41,
            'fold': fold,
            'early_stopping_patience': 5,
            'early_stopping_min_delta': 1e-4,
            'scheduler': 'StepLR',           # NEW
            'scheduler_step_size': 5,         # NEW
            'scheduler_gamma': 0.5,           # NEW
        }
    )
    wandb.watch(model, criterion, log='gradients', log_freq=100)

    # Siapkan folder output untuk confusion matrix
    cm_dir = os.path.join(os.getcwd(), 'confusion_matrix')
    os.makedirs(cm_dir, exist_ok=True)

    # Siapkan folder output untuk checkpoints
    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints', f'fold{fold+1}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # NEW: Inisialisasi Early Stopping (monitor val_loss)
    early_patience = wandb.config.early_stopping_patience
    early_min_delta = wandb.config.early_stopping_min_delta
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training Loop
    epochs = wandb.config.epochs
    best_val_acc = 0.0
    best_epoch = 0
    start_epoch = 0

    # Resume dari checkpoint jika diminta
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        checkpoint = load_checkpoint(resume_from_checkpoint, model, optimizer, scheduler)
        if checkpoint:
            start_epoch = checkpoint.get('epoch', 0)
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            best_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            print(f"Melanjutkan training dari epoch {start_epoch + 1}")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        for specs, labels, _ in train_loader:
            specs = specs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * specs.size(0)

        train_loss = running_loss / len(train_dataset)

        # Validation Loop
        model.eval()
        val_running_loss = 0.0
        all_labels, all_preds = [], []
        with torch.no_grad():
            for specs, labels, _ in val_loader:
                specs = specs.to(device)
                labels = labels.to(device).long()

                outputs = model(specs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * specs.size(0)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds)

        val_loss = val_running_loss / len(val_dataset)

        # CHANGED: step scheduler tanpa argumen (StepLR tidak perlu metrik)
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != prev_lr:
            print(f"[Fold {fold+1}] LR reduced: {prev_lr:.6f} -> {new_lr:.6f}")

        # Confusion matrix dan metrik
        cm = confusion_matrix(all_labels, all_preds)

        with np.errstate(divide='ignore', invalid='ignore'):
            tp = np.diag(cm).astype(float)
            fp = cm.sum(axis=0) - tp
            fn = cm.sum(axis=1) - tp

            precision_per_class = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
            recall_per_class = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
            f1_per_class = np.divide(2 * precision_per_class * recall_per_class,
                                     precision_per_class + recall_per_class,
                                     out=np.zeros_like(tp),
                                     where=(precision_per_class + recall_per_class) != 0)

        precision = float(np.nanmean(precision_per_class))
        recall = float(np.nanmean(recall_per_class))
        acc_cm = (tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0
        f1 = float(np.nanmean(f1_per_class))

        print(f"[Fold {fold+1}] Epoch [{epoch+1}/{epochs}] "
              f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
              f"acc: {acc_cm:.4f} | P/R/F1: {precision:.4f}/{recall:.4f}/{f1:.4f}")

        # Visualisasi confusion matrix
        with np.errstate(all='ignore'):
            cm_norm = cm.astype(float)
            row_sums = cm_norm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel='True label',
            xlabel='Predicted label',
            title=f'Confusion Matrix (normalized) - Fold {fold+1} - Epoch {epoch+1}'
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = 0.5
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm_norm[i, j] if not np.isnan(cm_norm[i, j]) else 0.0
                ax.text(j, i, f"{val:.2f}\n({cm[i, j]})",
                        ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=8)

        fig.tight_layout()
        img_path = os.path.join(cm_dir, f'confusion_matrix_fold{fold+1}_epoch_{epoch+1}.png')
        fig.savefig(img_path, bbox_inches='tight')
        wandb.log({'val/confusion_matrix_image': wandb.Image(fig)}, step=epoch + 1)
        wandb.save(img_path)
        plt.close(fig)

        # Log ke W&B
        log_dict = {
            'epoch': epoch + 1,
            'fold': fold + 1,  # 1-based untuk kemudahan baca
            'train/loss': train_loss,
            'val/loss': val_loss,
            'val/accuracy': acc_cm,
            'val/precision': precision,
            'val/recall': recall,
            'val/f1': f1,
            'lr': optimizer.param_groups[0]['lr'],  # akan mencerminkan LR terbaru setelah scheduler.step()
        }
        wandb.log(log_dict, step=epoch + 1)

        # Simpan checkpoint setiap epoch
        checkpoint = {
            'epoch': epoch + 1,
            'fold': fold + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': acc_cm,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'best_val_acc': best_val_acc,
            'config': dict(wandb.config),
        }
        
        # Simpan checkpoint epoch terakhir
        last_checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
        torch.save(checkpoint, last_checkpoint_path)
        
        # Simpan checkpoint setiap epoch dengan nama berbeda (opsional, bisa dinonaktifkan jika menghemat space)
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, epoch_checkpoint_path)

        # Simpan model terbaik per fold (berdasarkan akurasi)
        if acc_cm > best_val_acc:
            best_val_acc = acc_cm
            best_epoch = epoch + 1
            
            # Simpan di root directory (backward compatibility)
            best_model_path = f'best_model_fold{fold+1}.pth'
            torch.save(checkpoint, best_model_path)
            wandb.save(best_model_path)
            
            # Simpan juga di folder checkpoints
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_checkpoint_path)
            
            print(f"[Fold {fold+1}] ✓ New best model saved! Epoch {epoch+1}, Val Acc: {acc_cm:.4f}")

        # NEW: Early Stopping check (monitor val_loss)
        if val_loss < best_val_loss - early_min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"[Fold {fold+1}] EarlyStopping counter: {epochs_no_improve}/{early_patience}")
            if epochs_no_improve >= early_patience:
                print(f"[Fold {fold+1}] Early stopping triggered at epoch {epoch+1}.")
                break

    # Summary setelah training selesai
    print(f"\n{'='*60}")
    print(f"Training Fold {fold+1} completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f} (at epoch {best_epoch})")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print(f"{'='*60}\n")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=-1,
                        help="Pilih fold 1..5. Gunakan -1 untuk menjalankan semua fold berurutan.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path ke checkpoint untuk melanjutkan training (contoh: checkpoints/fold1/last_checkpoint.pth)")
    args = parser.parse_args()

    if 1 <= args.fold <= 5:
        # Jalankan satu fold (user 1..5 → indeks 0..4)
        run_train(args.fold - 1, resume_from_checkpoint=args.resume)
    else:
        # Jalankan semua fold 1..5
        for f in range(5):
            run_train(f)
