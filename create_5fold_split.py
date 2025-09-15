import os
import json
import random

def make_5fold_split(samples, n_folds=5, seed=42):
    # samples: list of (file_path, label)
    samples = [(os.path.abspath(fpath), label) for fpath, label in samples]
    random.seed(seed)
    random.shuffle(samples)
    fold_size = len(samples) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(samples)
        val = samples[start:end]
        train = samples[:start] + samples[end:]
        folds.append({
            "train": [{"file_path": fp, "label": lbl} for fp, lbl in train],
            "val": [{"file_path": fp, "label": lbl} for fp, lbl in val]
        })
    return folds

if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), 'data')
    labels_map = {'ambulance': 0, 'car_horn': 1}
    samples = []
    for label in labels_map:
        label_dir = os.path.join(data_dir, label)
        for fname in os.listdir(label_dir):
            fpath = os.path.join(label_dir, fname)
            if fname.endswith('.wav') and os.path.isfile(fpath):
                samples.append((fpath, label))
    folds = make_5fold_split(samples, n_folds=5)
    split_path = os.path.join(os.getcwd(), "split.json")
    with open(split_path, "w") as f:
        json.dump(folds, f, indent=2)
    print(f"Saved 5-fold split to {split_path}")
