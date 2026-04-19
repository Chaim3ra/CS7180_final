import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CSVDataset(Dataset):
    """PyTorch Dataset for loading CSV files from data/raw/.

    Args:
        csv_path (str): Path to the CSV file.
        target_col (str): Name of the target/label column.
        transform (callable, optional): Transform applied to feature tensors.
    """

    def __init__(self, csv_path: str, target_col: str, transform=None):
        self.df = pd.read_csv(csv_path)
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in {csv_path}")

        self.targets = torch.tensor(self.df[target_col].values, dtype=torch.float32)
        self.features = torch.tensor(
            self.df.drop(columns=[target_col]).values, dtype=torch.float32
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = self.targets[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def get_dataloader(
    csv_path: str,
    target_col: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    transform=None,
) -> DataLoader:
    """Convenience factory that returns a DataLoader for a CSV file.

    Args:
        csv_path (str): Path to the CSV file (typically under data/raw/).
        target_col (str): Name of the target/label column.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle samples each epoch.
        num_workers (int): Subprocesses for data loading (0 = main process).
        transform (callable, optional): Transform applied to feature tensors.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    dataset = CSVDataset(csv_path, target_col, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    # Quick smoke-test: place any CSV in data/raw/ and update these values.
    RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    example_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]

    if not example_files:
        print("No CSV files found in data/raw/. Add one to test the dataloader.")
    else:
        path = os.path.join(RAW_DIR, example_files[0])
        print(f"Loading: {path}")
        # Replace 'label' with your actual target column name.
        loader = get_dataloader(path, target_col="label", batch_size=4)
        features, targets = next(iter(loader))
        print(f"Feature shape: {features.shape}  |  Target shape: {targets.shape}")
