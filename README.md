# Klasifikasi Suara Lingkungan dengan PiczakCNN

Sistem klasifikasi audio lingkungan untuk tugas akhir menggunakan arsitektur PiczakCNN, fitur log-Mel + delta, augmentasi audio, dan evaluasi 5-fold cross-validation.

---

## Daftar Isi
- [Requirements](#requirements)
- [Instalasi](#instalasi)
- [Struktur Folder](#struktur-folder)
- [Cara Menjalankan](#cara-menjalankan)
- [Pipeline Preprocessing](#pipeline-preprocessing)
- [Troubleshooting](#troubleshooting)
- [How to Cite](#how-to-cite)

---

## Requirements

| Komponen | Versi |
|----------|-------|
| Python | 3.8+ (disarankan 3.10+) |
| PyTorch | 2.7.1 |
| Torchaudio | 2.7.1 |
| CUDA | Opsional (GPU acceleration) |

Spesifikasi minimum:
- RAM 8 GB (disarankan 16 GB)
- GPU NVIDIA opsional untuk training lebih cepat

---

## Instalasi

### 1. Clone dan Buat Virtual Environment

```bash
git clone <url-repository>
cd audio_classification

# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Opsional: Integrasi Weights and Biases

```bash
wandb login
```


---

## Cara Menjalankan

Pastikan folder data sudah berisi file audio berformat wav di dalam subfolder kelas.

### 1. Generate Split 5-Fold

```bash
python create_5fold_split.py
```

### 2. Training Semua Fold

```bash
python train.py --fold -1
```

### 3. Training Satu Fold Saja

```bash
python train.py --fold 1
```

### 4. Resume dari Checkpoint

```bash
python train.py --fold 1 --resume checkpoints/fold1/last_checkpoint.pth
```

### 5. Jalankan Ablation Study

```bash
python ablation.py --fold -1
```

### 6. Verifikasi Environment

```bash
python -c "import torch, torchaudio, numpy, sklearn, matplotlib, wandb, torchinfo; print('OK')"
```

---

## Pipeline Preprocessing

```text
Audio WAV
	-> Load waveform (torchaudio)
	-> (Train only) Time Shift
	-> (Train only) Add Background Noise
	-> Mel Spectrogram (n_mels=60)
	-> Amplitude to dB (log-Mel)
	-> Pad/Crop ke segment 41 frame
	-> (Train only) SpecAugment
	-> Delta Feature
	-> Stack [log-Mel, delta] -> input model
```

Konfigurasi penting:
- sample_rate: 44100
- n_fft: 1024
- hop_length: 512
- n_mels: 60
- segment_length: 41

---

## Dataset

Kelas suara yang digunakan:
- klakson_mobil
- klakson_motor
- pecahan_kaca
- sirine
- tangisan_bayi

Format data:
- Satu folder per kelas di dalam folder data
- File audio berformat wav

---

## Model

Arsitektur utama:
- Backbone: PiczakCNN
- Input shape: (60, 41)
- Training default: SGD (lr=0.002, momentum=0.9, nesterov=True)
- Scheduler: ReduceLROnPlateau
- Evaluasi: Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

## Troubleshooting

| Error | Solusi |
|-------|--------|
| No module named torch | pip install -r requirements.txt |
| CUDA out of memory | Kurangi batch size atau gunakan CPU |
| FileNotFoundError pada data | Pastikan struktur folder data per kelas sudah benar |
| Checkpoint tidak ditemukan | Periksa path resume, contoh checkpoints/fold1/last_checkpoint.pth |
| Split bermasalah | Jalankan ulang python create_5fold_split.py |

---

## How to Cite

Jika repositori ini digunakan pada riset atau publikasi, mohon cantumkan sitasi tugas akhir Anda.



**Format IEEE:**
> I. J. B. Regen, "Klasifikasi Suara Lingkungan Menggunakan Piczak CNN Berdasarkan Representasi Spektrogram," Skripsi, Program Studi Teknik Informatika, Institut Teknologi Sumatera, 2026.

**Format APA:**
> Regen, I. J. B. (2026). *Klasifikasi Suara Lingkungan Menggunakan Piczak CNN Berdasarkan Representasi Spektrogram* (Skripsi). Institut Teknologi Sumatera.

**BibTeX:**
```bibtex
@mastersthesis{regen2026klasifikasi,
  author       = {Regen, Ignatius Julio Bintang},
  title        = {Klasifikasi Suara Lingkungan Menggunakan Piczak CNN Berdasarkan Representasi Spektrogram},
  school       = {Institut Teknologi Sumatera},
  year         = {2026},
  type         = {Skripsi}
}

---

Version: 1.0.0  
Status: Production Ready  
Last Updated: April 2026
