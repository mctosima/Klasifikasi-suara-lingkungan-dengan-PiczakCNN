import os   
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import json
from create_5fold_split import make_5fold_split
import random
from utils import preview_mel_spectrogram
import warnings
import torch.nn.functional as F  # <— diperlukan untuk F.pad
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

def time_shift_tensor(waveform: torch.Tensor, max_shift_ratio: float = 0.2) -> torch.Tensor:
    """
    waveform: Tensor [C, N]
    """
    if max_shift_ratio <= 0 or waveform.size(1) < 2:
        return waveform
    # random.random()*2 - 1 -> range (-1,1)
    shift = int(waveform.size(1) * max_shift_ratio * (random.random() * 2 - 1))
    if shift == 0:
        return waveform
    if shift > 0:
        pad = torch.zeros(waveform.size(0), shift, dtype=waveform.dtype)
        waveform = torch.cat([pad, waveform[:, :-shift]], dim=1)
    else:
        shift = -shift
        pad = torch.zeros(waveform.size(0), shift, dtype=waveform.dtype)
        waveform = torch.cat([waveform[:, shift:], pad], dim=1)
    return waveform

def add_background_noise_tensor(waveform: torch.Tensor, snr_db_range=(15, 25)) -> torch.Tensor:
    """
    Menambahkan white noise pada tiap channel dengan SNR acak dalam rentang snr_db_range.
    waveform: Tensor [C, N]
    snr_db_range: tuple (min_dB, max_dB)
    """
    C, N = waveform.shape
    if N == 0:
        return waveform
    snr_db = np.random.uniform(snr_db_range[0], snr_db_range[1])
    signal_power = waveform.pow(2).mean().item()
    if signal_power <= 1e-12:
        return waveform
    noise_power = signal_power / (10 ** (snr_db / 10))
    # Generate Gaussian noise per channel
    noise = torch.randn_like(waveform)
    current_noise_power = noise.pow(2).mean().item()
    if current_noise_power <= 1e-12:
        return waveform
    scale = (noise_power / current_noise_power) ** 0.5
    waveform_noisy = waveform + noise * scale
    return waveform_noisy

def spec_augment_tensor(mel: torch.Tensor,
                        freq_mask_param: int = 8,
                        time_mask_param: int = 10,
                        num_freq_masks: int = 1,
                        num_time_masks: int = 1) -> torch.Tensor:
    """
    mel: Tensor [C, n_mels, T]
    Zero-mask acak pada dimensi frekuensi dan waktu (SpecAugment sederhana).
    """
    C, Freq, Time = mel.shape
    # Frequency masks
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        if f > 0 and f < Freq:
            f0 = random.randint(0, Freq - f)
            mel[:, f0:f0+f, :] = 0
    # Time masks
    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        if t > 0 and t < Time:
            t0 = random.randint(0, Time - t)
            mel[:, :, t0:t0+t] = 0
    return mel

class AudioDataset(Dataset):
    def __init__(self, root_dir, fold=0, split_json="split.json", split_type="train", segment_length=41, apply_augment=True):
        self.labels_map={
            'ambulance': 0,
            'klakson_mobil': 1,
        }

        self.segment_length = segment_length

        # --SELF HANDLING--
        split_path = os.path.join(os.getcwd(), split_json)

        # Jika file json ada
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                self.splits = json.load(f)

        # Jika file json tidak ada
        else:
            all_samples = []
            for label in self.labels_map:
                label_dir = os.path.join(root_dir, label)
                for fname in os.listdir(label_dir):
                    fpath = os.path.join(label_dir, fname)
                    if fname.endswith('.wav') and os.path.isfile(fpath):
                        all_samples.append((fpath, label))
        
        # Panggil fugnsi untuk membuat split 5-fold
            folds = make_5fold_split(all_samples, n_folds=5)
            with open(split_path, "w") as f:
                json.dump(folds, f, indent=2)
            self.splits = folds


        # --END SELF HANDLING--


        # Mempersiapkan konfigurasi fitur
        self.mel_spectogram = T.MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=512,
            n_mels=60,
        )

        self.delta_transform = T.ComputeDeltas()
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

        # Memuat daftar sample akhirnya
        self.samples = [(item["file_path"], item["label"]) for item in self.splits[fold][split_type]]
        self.apply_augment = (split_type == "train") and apply_augment
        self.noise_snr_range = (15, 25)   # rentang SNR dB
        self.time_shift_ratio = 0.2
        # Semua augmentasi menggunakan probabilitas tetap 50%
        self.specaug_freq_param = 8
        self.specaug_time_param = 10
        self.specaug_num_freq_masks = 1
        self.specaug_num_time_masks = 1


    def __len__(self):
        """mengembalikan jumlah sampel dalam dataset"""

        return len(self.samples)

    def __getitem__(self, idx):
        """mengambil item dari dataset berdasarkan indeks"""

        file_path, label = self.samples[idx]

        # Cek apakah file ada, jika tidak ada akan menampilkan eror
        if not os.path.isfile(file_path):
            print(f"File tidak ada: {file_path}")
            return file_path, label, None
        
        try:
            waveform, sr = torchaudio.load(file_path)  # Memuat file audio

            # Augmentasi secara serial dengan probabilitas 50% untuk setiap tahap
            if self.apply_augment:
                # Augmentasi 1: Time shift (probabilitas 50%)
                if random.random() < 0.5:
                    waveform = time_shift_tensor(waveform, max_shift_ratio=self.time_shift_ratio)
                
                # Augmentasi 2: Noise injection (probabilitas 50%)
                if random.random() < 0.5:
                    waveform = add_background_noise_tensor(waveform, snr_db_range=self.noise_snr_range)

            # Ekstraksi fitur
            mel_spec = self.mel_spectogram(waveform)
            log_mel_spec = self.amplitude_to_db(mel_spec)

            # Padding bila kurang panjang
            if log_mel_spec.shape[2] < self.segment_length:
                padding = self.segment_length - log_mel_spec.shape[2]
                log_mel_spec = F.pad(log_mel_spec, (0, padding))

            # Crop acak
            start_frame = random.randint(0, log_mel_spec.shape[2] - self.segment_length)
            segment = log_mel_spec[:, :, start_frame:start_frame + self.segment_length]

            # Augmentasi 3: SpecAugment (probabilitas 50%)
            if self.apply_augment:
                if random.random() < 0.5:
                    segment = spec_augment_tensor(
                        segment.clone(),  # clone agar aman
                        freq_mask_param=self.specaug_freq_param,
                        time_mask_param=self.specaug_time_param,
                        num_freq_masks=self.specaug_num_freq_masks,
                        num_time_masks=self.specaug_num_time_masks
                    )

            # Delta
            deltas = self.delta_transform(segment)
            stacked_features = torch.cat((segment, deltas), dim=0)
            return file_path, label, stacked_features
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return file_path, label, None

   
    
if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), 'data')
    dataset = AudioDataset(root_dir=data_dir, fold=0, split_json="split.json", split_type="train")

    print(f"Jumlah sampel : {len(dataset)}")

    random_idx = random.randint(0, len(dataset) - 1)

    sample_data = dataset[random_idx]
    file_path, label, features = sample_data
    print(f"File Path: {file_path}, Label: {label}")
    print(f"Stacked features shape: {features.shape if features is not None else 'None'}")

