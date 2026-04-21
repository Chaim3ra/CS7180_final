"""Polars-backed Dataset and LightningDataModule for solar generation forecasting.

All CSV I/O uses Polars for efficient loading of the large 1-min Pecan Street
and NSRDB files.  The Dataset creates sliding context/forecast windows over
aligned weather and generation time-series.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Union

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L  # type: ignore[no-redef]

__all__ = [
    "SolarWindowDataset",
    "SolarDataModule",
    "get_dataloader",
    "filter_solar_homes",
    "read_parquet",
    "read_csv",
    "write_parquet",
    "ensure_local_parquet",
]


def read_parquet(path: Union[str, Path]) -> pl.DataFrame:
    """Read a parquet file from local disk or directly from S3.

    If *path* starts with ``s3://``, the file is streamed from S3 via boto3
    into an in-memory buffer — no local disk write is required.  Otherwise
    the file is read from the local filesystem with Polars.

    Args:
        path: Local filesystem path or an S3 URI of the form
            ``s3://bucket/key/to/file.parquet``.

    Returns:
        Polars DataFrame containing the parquet contents.

    Raises:
        ImportError: If ``boto3`` is not installed and an S3 URI is given.
        FileNotFoundError: If a local path does not exist.
    """
    path = str(path)
    if path.startswith("s3://"):
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for S3 streaming. Install it with: pip install boto3"
            ) from exc
        # s3://bucket/key/to/file.parquet  ->  bucket="bucket", key="key/to/file.parquet"
        without_scheme = path[5:]
        bucket, _, key = without_scheme.partition("/")
        s3  = boto3.client("s3")
        buf = io.BytesIO()
        s3.download_fileobj(bucket, key, buf)
        buf.seek(0)
        return pl.read_parquet(buf)
    return pl.read_parquet(path)


def read_csv(path: Union[str, Path], **kwargs) -> pl.DataFrame:
    """Read a CSV file from local disk or directly from S3.

    If *path* starts with ``s3://``, the file is streamed from S3 via boto3
    into an in-memory buffer.  All keyword arguments are forwarded to
    :func:`polars.read_csv` (e.g. ``columns``, ``dtypes``).

    Args:
        path: Local filesystem path or ``s3://bucket/key`` URI.
        **kwargs: Forwarded to ``pl.read_csv``.

    Returns:
        Polars DataFrame.
    """
    path = str(path)
    if path.startswith("s3://"):
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for S3 streaming. Install it with: pip install boto3"
            ) from exc
        without_scheme = path[5:]
        bucket, _, key = without_scheme.partition("/")
        s3  = boto3.client("s3")
        buf = io.BytesIO()
        s3.download_fileobj(bucket, key, buf)
        buf.seek(0)
        return pl.read_csv(buf, **kwargs)
    return pl.read_csv(path, **kwargs)


def write_parquet(df: pl.DataFrame, path: Union[str, Path]) -> None:
    """Write a Polars DataFrame as parquet to local disk or S3.

    If *path* starts with ``s3://``, the parquet bytes are uploaded via
    boto3 without writing any local file.  Otherwise the file is written
    to the local filesystem (parent directories are created automatically).

    Args:
        df: DataFrame to serialise.
        path: Destination local path or ``s3://bucket/key`` URI.
    """
    path = str(path)
    if path.startswith("s3://"):
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for S3 uploads. Install it with: pip install boto3"
            ) from exc
        without_scheme = path[5:]
        bucket, _, key = without_scheme.partition("/")
        buf = io.BytesIO()
        df.write_parquet(buf)
        buf.seek(0)
        boto3.client("s3").upload_fileobj(buf, bucket, key)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)


def ensure_local_parquet(path: Union[str, Path], cache_dir: Union[str, Path]) -> str:
    """Return a local path to a parquet file, downloading from S3 if not cached.

    Checks ``cache_dir`` for a local copy first.  If the file is absent and
    ``path`` is an S3 URI, it is downloaded once to ``cache_dir`` via boto3.
    Subsequent calls reuse the local file without touching S3.

    Args:
        path: Local filesystem path or ``s3://bucket/key`` URI.
        cache_dir: Directory to store locally cached parquet files.

    Returns:
        Absolute local path string to the cached file.
    """
    path      = str(path)
    cache_dir = Path(cache_dir)

    if not path.startswith("s3://"):
        local = Path(path)
        if local.exists():
            print(f"  Using local cache: {local.name}")
            return str(local)
        raise FileNotFoundError(f"Parquet not found: {local}")

    without_scheme = path[5:]
    bucket, _, key = without_scheme.partition("/")
    filename       = key.split("/")[-1]
    local          = cache_dir / filename

    if local.exists():
        print(f"  Using local cache: {local.name}")
        return str(local)

    try:
        import boto3
    except ImportError as exc:
        raise ImportError("boto3 is required for S3 downloads. pip install boto3") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading from S3 to local cache: {filename} ...", flush=True)
    boto3.client("s3").download_file(bucket, key, str(local))
    size_mb = local.stat().st_size / 1024 ** 2
    print(f"  Downloaded: {local.name} ({size_mb:.1f} MB)")
    return str(local)


def filter_solar_homes(
    df: pl.DataFrame,
    solar_col: str = "solar_kwh",
    min_nonzero_frac: float = 0.05,
) -> list[int]:
    """Return home IDs that have sufficient non-zero solar generation.

    Homes where the solar column is all zeros or has fewer than
    ``min_nonzero_frac`` non-zero rows are excluded from training.
    Prints a summary line for every filtered home.

    Args:
        df: DataFrame containing ``dataid`` and ``solar_col`` columns.
        solar_col: Name of the solar generation column.
        min_nonzero_frac: Minimum fraction of non-zero rows required.

    Returns:
        Sorted list of dataid integers that pass the filter.
    """
    valid: list[int] = []
    home_ids = df["dataid"].unique().sort().to_list()
    for dataid in home_ids:
        home = df.filter(pl.col("dataid") == dataid)
        total   = home.height
        nonzero = home.filter(pl.col(solar_col) > 0).height
        frac    = nonzero / total if total > 0 else 0.0
        if frac < min_nonzero_frac:
            reason = "all zeros" if nonzero == 0 else f"{frac:.1%} non-zero < {min_nonzero_frac:.0%}"
            print(f"  [FILTERED] dataid={dataid:>5}  {reason}  ({nonzero}/{total} rows)")
        else:
            valid.append(dataid)
    return valid


class SolarWindowDataset(Dataset):
    """Sliding-window Dataset for multi-modal solar generation forecasting.

    Loads weather and generation CSVs via Polars (lazy scan for memory
    efficiency), then creates non-overlapping windows of length ``seq_len``
    followed by a forecast target of length ``forecast_horizon``.

    Each ``__getitem__`` call returns a tuple of four tensors:

    .. code-block:: text

        weather    (seq_len, len(weather_cols))   float32
        generation (seq_len, 1)                   float32
        metadata   (len(metadata),)               float32
        target     (forecast_horizon,)            float32  ← next kWh values

    Args:
        weather_path: Path to a CSV containing weather time-series columns.
        generation_path: Path to a CSV containing the generation column.
        weather_cols: Column names to select from ``weather_path``.
        generation_col: Name of the kWh column in ``generation_path``.
        metadata: Static site features as a plain Python list (lat, lon, tilt,
            azimuth, capacity_kw, elevation_m).
        seq_len: Context window length (number of time steps fed to encoders).
        forecast_horizon: Number of future steps to predict.
    """

    def __init__(
        self,
        weather_path: Union[str, Path],
        generation_path: Union[str, Path],
        weather_cols: list[str],
        generation_col: str,
        metadata: list[float],
        seq_len: int = 96,
        forecast_horizon: int = 4,
    ):
        weather_arr = (
            pl.scan_csv(weather_path)
            .select(weather_cols)
            .collect()
            .to_numpy()
        )
        gen_arr = (
            pl.scan_csv(generation_path)
            .select([generation_col])
            .collect()
            .fill_null(0.0)
            .to_numpy()
        )

        n_weather = len(weather_arr)
        n_gen = len(gen_arr)
        n = min(n_weather, n_gen)

        self.weather = torch.tensor(weather_arr[:n], dtype=torch.float32)
        self.generation = torch.tensor(gen_arr[:n], dtype=torch.float32)
        self.metadata = torch.tensor(metadata, dtype=torch.float32)
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon

    def __len__(self) -> int:
        return max(0, len(self.weather) - self.seq_len - self.forecast_horizon + 1)

    def __getitem__(self, idx: int):
        """Return one (weather, generation, metadata, target) window.

        Args:
            idx: Start index of the context window.

        Returns:
            Tuple of four float32 tensors with shapes:
            ``(seq_len, W)``, ``(seq_len, 1)``, ``(M,)``, ``(H,)``.
        """
        end = idx + self.seq_len
        weather_window = self.weather[idx:end]
        gen_window = self.generation[idx:end]
        target = self.generation[end : end + self.forecast_horizon, 0]
        return weather_window, gen_window, self.metadata, target


class SolarDataModule(L.LightningDataModule):
    """LightningDataModule that splits a single site's data into train/val/test.

    Reads ``data`` and ``trainer`` sections from the experiment config dict and
    builds three :class:`SolarWindowDataset` instances by slicing the full
    time-series chronologically.

    Args:
        cfg: Full experiment config dict (loaded from ``experiment.yaml``).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["data"]
        self.weather_path = d["weather_path"]
        self.generation_path = d["generation_path"]
        self.weather_cols = d["weather_cols"]
        self.generation_col = d["generation_col"]
        self.metadata = d["metadata"]
        self.seq_len = d["seq_len"]
        self.forecast_horizon = d["forecast_horizon"]
        self.train_frac = d["train_frac"]
        self.val_frac = d["val_frac"]
        self.batch_size = d["batch_size"]
        self.num_workers = d["num_workers"]

        self._train_ds: Optional[SolarWindowDataset] = None
        self._val_ds:   Optional[SolarWindowDataset] = None
        self._test_ds:  Optional[SolarWindowDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and build train/val/test datasets.

        Args:
            stage: ``"fit"``, ``"validate"``, ``"test"``, or ``None`` (all).
        """
        weather_full = (
            pl.scan_csv(self.weather_path)
            .select(self.weather_cols)
            .collect()
            .to_numpy()
        )
        gen_full = (
            pl.scan_csv(self.generation_path)
            .select([self.generation_col])
            .collect()
            .fill_null(0.0)
            .to_numpy()
        )
        n = min(len(weather_full), len(gen_full))

        train_end = int(n * self.train_frac)
        val_end   = int(n * (self.train_frac + self.val_frac))

        def _make(start: int, end: int) -> SolarWindowDataset:
            ds = SolarWindowDataset.__new__(SolarWindowDataset)
            ds.weather    = torch.tensor(weather_full[start:end], dtype=torch.float32)
            ds.generation = torch.tensor(gen_full[start:end],     dtype=torch.float32)
            ds.metadata   = torch.tensor(self.metadata, dtype=torch.float32)
            ds.seq_len         = self.seq_len
            ds.forecast_horizon = self.forecast_horizon
            return ds

        if stage in ("fit", None):
            self._train_ds = _make(0, train_end)
            self._val_ds   = _make(train_end, val_end)
        if stage in ("validate", None):
            self._val_ds   = _make(train_end, val_end)
        if stage in ("test", None):
            self._test_ds  = _make(val_end, n)

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def get_dataloader(
    weather_path: Union[str, Path],
    generation_path: Union[str, Path],
    weather_cols: list[str],
    generation_col: str,
    metadata: list[float],
    seq_len: int = 96,
    forecast_horizon: int = 4,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Convenience factory that returns a DataLoader for a single site.

    Args:
        weather_path: Path to the weather CSV file.
        generation_path: Path to the generation CSV file.
        weather_cols: Column names to select from the weather CSV.
        generation_col: Name of the kWh column in the generation CSV.
        metadata: Static site features (lat, lon, tilt, azimuth, capacity, elev).
        seq_len: Context window length.
        forecast_horizon: Number of steps to predict.
        batch_size: Samples per batch.
        shuffle: Whether to shuffle samples each epoch.
        num_workers: Subprocesses for data loading (0 = main process).

    Returns:
        Configured :class:`~torch.utils.data.DataLoader`.
    """
    dataset = SolarWindowDataset(
        weather_path=weather_path,
        generation_path=generation_path,
        weather_cols=weather_cols,
        generation_col=generation_col,
        metadata=metadata,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
