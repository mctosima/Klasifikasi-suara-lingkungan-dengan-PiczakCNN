import torch

def preview_mel_spectrogram(file_path, label, mel_spec):
    if mel_spec is not None:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec.squeeze().numpy(), aspect='auto', origin='lower')
        plt.title(f"Mel Spectrogram - {label}")
        plt.xlabel("Frames")
        plt.ylabel("Mel bins")
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Cannot preview Mel spectrogram for {file_path} due to loading error.")

def check_set_gpu(override=None):
    if override == None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"Using MPS: {torch.backends.mps.is_available()}")
        else:
            device = torch.device('cpu')
            print(f"Using CPU: {torch.device('cpu')}")
    else:
        device = torch.device(override)
    return device

def collate_fn(batch):
    """
    Custom collate function to handle potential loading errors and stack tensors.
    The datareader now provides the correct 2-channel tensor, so no more channel duplication.
    """
    # Filter out samples that failed to load
    batch = [b for b in batch if b[2] is not None]
    if not batch:
        return None, None, None
    paths, labels, specs = zip(*batch)
    
    # Stack the 2-channel spectrograms from the batch
    specs = torch.stack(specs)
    # Convert string labels to numerical tensors for binary classification
    labels = torch.tensor([0 if l == "ambulance" else 1 for l in labels], dtype=torch.float32)
    return specs, labels, paths