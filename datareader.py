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
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

class AudioDataset(Dataset):
    def __init__(self, root_dir, fold=0, split_json="split.json", split_type="train" ,segment_length=41):
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
            waveform, sr = torchaudio.load(file_path) # Memuat file audio

            ## 1.Ekstrasi fitur Mel Spectrogram
            mel_spec = self.mel_spectogram(waveform)
            log_mel_spec = self.amplitude_to_db(mel_spec)

            ## 2. Segmentasi
            ## Jika spectogram lebih pendek dari segment, maka kita padding
            if log_mel_spec.shape[2] < self.segment_length:
                padding = self.segment_length - log_mel_spec.shape[2]
                log_mel_spec = F.pad(log_mel_spec, (0, padding))
            
            ## 3. Pilih starting frame secara acak untuk segment
            start_frame = random.randint(0, log_mel_spec.shape[2] - self.segment_length)
            segment = log_mel_spec[:, :, start_frame:start_frame + self.segment_length]

            ## 4. Hitung delta
            deltas = self.delta_transform(segment)
            # print(f"Delta: {deltas}")
            # print(f"Delta shape: {deltas.shape}")

            ## 5. Gabungkan segment dan delta
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

    # Preview mel spectrogram
    preview_mel_spectrogram(file_path, label, features[0, :, :].unsqueeze(0))
