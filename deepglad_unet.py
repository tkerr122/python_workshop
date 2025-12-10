#!/usr/bin/env python3
# ============================================================
#   watch -n0.5 nvidia-smi
#   ssh gladapp20
#   module load gpu/python/3.7/anaconda
#   module load python/3.7/anaconda
#   source activate city2021
#   cd v03/deeplearning
#   module unload rh9/gdal/3.11.0
# ============================================================

# =========================
# Description
# =========================
##The script implements an end-to-end binary segmentation pipeline for GeoTIFF mosaics. 
##It defines a U-Net in TensorFlow/Keras (using MirroredStrategy when multiple GPUs are available) 
##that ingests per-tile metric stacks (one folder per metric, one .tif per tile), normalizes labels to {0,1}, 
##and generates non-overlapping training chips. Before the train/validation split, it applies configurable offline 
##augmentation via 90°/180°/270° rotations and optional horizontal/vertical flips to increase data diversity 
##without interpolation artifacts. Training uses standard callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint) 
##and saves the best weights. For inference, the model slides over the full image with overlap and blends window predictions 
##using a 2D Hann mask to suppress seam artifacts; probabilities are scaled to 0–100 and written as compressed (LZW) GeoTIFFs, 
##copying geotransform and projection from a reference raster. The code also validates that all required metric files exist for 
##each tile, logs any missing tiles, and organizes outputs by year and directory.

from __future__ import annotations

import os
import glob
import gc
import logging
from typing import List, Tuple, Dict, Optional, Union

import numpy as np

try:
    from osgeo import gdal, osr  # type: ignore
except Exception:
    import gdal  # type: ignore
    import osr   # type: ignore

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import callbacks as kcb
from tensorflow.keras import layers as L
from tensorflow.keras import Model

# Utils
from sklearn.model_selection import train_test_split


# =========================
# Global configuration
# =========================
SEED = 195
np.random.seed(SEED)
tf.random.set_seed(SEED)

# === Metrics order: one folder per metric name, one .tif per tile ===
METRICS_ORDER: List[str] = [
    "red_max",
    "green_avmin25_S2N",
    "red_av75max_LST",
    "blue_av75max_LST",
]
N_METRICS = len(METRICS_ORDER)

GRID = 256                 # chip size (square)
N_EPOCHS = 500
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_PRED = 16
OUTPUT_SCALE = 100.0       # scale prob [0,1] to [0,100] before saving as Byte
NODATA_LABEL = 255

# Let TF grow GPU memory as needed (TF 2.x)
try:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

logging.getLogger("tensorflow").setLevel(logging.FATAL)


# =========================
# Filesystem utilities
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_missing(metrics_root: str, tile_id: str, metrics_order: Optional[List[str]] = None) -> Optional[str]:
    """
    Check {metrics_root}/{metric}/{tile_id}.tif exists for each required metric.
    Returns the first missing metric name or None if everything is present.
    """
    if metrics_order is None:
        metrics_order = METRICS_ORDER
    for m in metrics_order:
        p = os.path.join(metrics_root, m, f"{tile_id}.tif")
        if not os.path.isfile(p):
            return m
    return None


# =========================
# Reading data
# =========================
def read_metrics_stack(metrics_root: str,
                       tile_id: str,
                       metrics_order: Optional[List[str]] = None) -> Dict:
    """
    Read and stack bands for a tile:
      {metrics_root}/{metric}/{tile_id}.tif
    Returns a dict with:
      - data: (B, rows, cols) float32
      - nbands, rows, cols, bandnames
      - ref_path: path to the first raster (to georeference outputs)
    """
    if metrics_order is None:
        metrics_order = METRICS_ORDER

    bands = []
    ref_path = None
    rows = cols = None

    for i, name in enumerate(metrics_order):
        path = os.path.join(metrics_root, name, f"{tile_id}.tif")
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            raise FileNotFoundError(f"Missing metric '{name}' for tile '{tile_id}': {path}")

        arr = ds.ReadAsArray()  # (rows, cols)
        if arr is None:
            raise RuntimeError(f"Could not read data from {path}")

        if i == 0:
            rows, cols = arr.shape
            ref_path = path
        else:
            if arr.shape != (rows, cols):
                raise ValueError(f"Shape mismatch in {path}: {arr.shape} vs {(rows, cols)}")

        bands.append(arr.astype(np.float32))

    data = np.stack(bands, axis=0).astype(np.float32)  # (B, rows, cols)
    return {
        "tile_id": tile_id,
        "ref_path": ref_path,
        "data": data,
        "nbands": data.shape[0],
        "rows": data.shape[1],
        "cols": data.shape[2],
        "bandnames": metrics_order,
        "nfiles": len(metrics_order),
    }


def read_label_raster(label_path: str) -> Dict:
    """
    Read a label raster and normalize to {0,1}:
      - NoData (255) -> 0
      - >2 -> 0
      - 1 or 2 -> 1
    Returns dict with data (1, rows, cols), nbands=1, etc.
    """
    ds = gdal.Open(label_path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Could not open label: {label_path}")

    data = ds.ReadAsArray()
    if data.ndim == 2:
        data = data.copy()
        data[data == NODATA_LABEL] = 0
        data[data > 2] = 0
        data[(data == 1) | (data == 2)] = 1
        data = np.stack([data], axis=0)  # (1, rows, cols)

    return {
        "filename": os.path.basename(label_path),
        "data": data.astype(np.float32),
        "nbands": 1,
        "rows": data.shape[1],
        "cols": data.shape[2],
    }


# =========================
# Sliding window utilities
# =========================
def compute_grid_indices(rows: int, cols: int, chip: int, stride: int) -> List[Tuple[int, int]]:
    """
    Generate top-left (row, col) indices for each chip, adjusting borders
    so every window is exactly 'chip' sized.
    """
    starts_r = list(range(0, rows, stride))
    starts_c = list(range(0, cols, stride))

    if not starts_r:
        starts_r = [0]
    if not starts_c:
        starts_c = [0]

    # Ensure the last chip touches the border
    if starts_r[-1] + chip < rows:
        starts_r.append(rows - chip)
    if starts_c[-1] + chip < cols:
        starts_c.append(cols - chip)

    # Clamp to valid range (in case image < chip)
    starts_r = [max(0, min(sr, rows - chip)) for sr in starts_r]
    starts_c = [max(0, min(sc, cols - chip)) for sc in starts_c]

    # Unique pairs
    pairs: List[Tuple[int, int]] = []
    seen = set()
    for sr in starts_r:
        for sc in starts_c:
            if (sr, sc) not in seen:
                pairs.append((sr, sc))
                seen.add((sr, sc))
    return pairs


def hann_weight_2d(chip: int) -> np.ndarray:
    """
    2D Hann weight mask of size chip x chip for seamless blending.
    """
    w1d = np.hanning(chip)
    w2d = np.outer(w1d, w1d)
    w2d = (w2d / max(w2d.max(), 1e-6)).astype(np.float32)  # [0,1]
    return w2d


# =========================
# Model (U-Net)
# =========================
def build_unet(n_metrics: int = N_METRICS) -> Model:
    """
    Simple U-Net with ELU activations and sigmoid output.
    Attempts to use MirroredStrategy if available.
    """
    try:
        strategy = tf.distribute.MirroredStrategy()
    except Exception:
        strategy = None

    def make() -> Model:
        inputs = L.Input((None, None, n_metrics))
        x = L.Lambda(lambda t: t / 255.0)(inputs)

        c1 = L.Conv2D(32, 3, padding="same", activation="elu", kernel_initializer="he_normal")(x)
        c1 = L.Dropout(0.1)(c1)
        c1 = L.Conv2D(32, 3, padding="same", activation="elu", kernel_initializer="he_normal")(c1)
        p1 = L.MaxPooling2D(2)(c1)

        c2 = L.Conv2D(64, 3, padding="same", activation="elu", kernel_initializer="he_normal")(p1)
        c2 = L.Dropout(0.1)(c2)
        c2 = L.Conv2D(64, 3, padding="same", activation="elu", kernel_initializer="he_normal")(c2)
        p2 = L.MaxPooling2D(2)(c2)

        c3 = L.Conv2D(128, 3, padding="same", activation="elu", kernel_initializer="he_normal")(p2)
        c3 = L.Dropout(0.2)(c3)
        c3 = L.Conv2D(128, 3, padding="same", activation="elu", kernel_initializer="he_normal")(c3)
        p3 = L.MaxPooling2D(2)(c3)

        c4 = L.Conv2D(256, 3, padding="same", activation="elu", kernel_initializer="he_normal")(p3)
        c4 = L.Dropout(0.2)(c4)
        c4 = L.Conv2D(256, 3, padding="same", activation="elu", kernel_initializer="he_normal")(c4)
        p4 = L.MaxPooling2D(2)(c4)

        c5 = L.Conv2D(512, 3, padding="same", activation="elu", kernel_initializer="he_normal")(p4)
        c5 = L.Dropout(0.3)(c5)
        c5 = L.Conv2D(512, 3, padding="same", activation="elu", kernel_initializer="he_normal")(c5)

        u6 = L.Conv2DTranspose(256, 2, strides=2, padding="same")(c5)
        u6 = L.Concatenate()([u6, c4])
        c6 = L.Conv2D(256, 3, padding="same", activation="elu", kernel_initializer="he_normal")(u6)
        c6 = L.Dropout(0.2)(c6)
        c6 = L.Conv2D(256, 3, padding="same", activation="elu", kernel_initializer="he_normal")(c6)

        u7 = L.Conv2DTranspose(128, 2, strides=2, padding="same")(c6)
        u7 = L.Concatenate()([u7, c3])
        c7 = L.Conv2D(128, 3, padding="same", activation="elu", kernel_initializer="he_normal")(u7)
        c7 = L.Dropout(0.2)(c7)
        c7 = L.Conv2D(128, 3, padding="same", activation="elu", kernel_initializer="he_normal")(c7)

        u8 = L.Conv2DTranspose(64, 2, strides=2, padding="same")(c7)
        u8 = L.Concatenate()([u8, c2])
        c8 = L.Conv2D(64, 3, padding="same", activation="elu", kernel_initializer="he_normal")(u8)
        c8 = L.Dropout(0.1)(c8)
        c8 = L.Conv2D(64, 3, padding="same", activation="elu", kernel_initializer="he_normal")(c8)

        u9 = L.Conv2DTranspose(32, 2, strides=2, padding="same")(c8)
        u9 = L.Concatenate()([u9, c1])
        c9 = L.Conv2D(32, 3, padding="same", activation="elu", kernel_initializer="he_normal")(u9)
        c9 = L.Dropout(0.1)(c9)
        c9 = L.Conv2D(8, 3, padding="same", activation="elu", kernel_initializer="he_normal")(c9)

        outputs = L.Conv2D(1, 1, activation="sigmoid")(c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    if strategy is not None:
        try:
            with strategy.scope():
                return make()
        except Exception:
            return make()
    else:
        return make()


# =========================
# GeoTIFF writing
# =========================
def write_geotiff_like(ref_path: str, out_path: str, array: np.ndarray, dtype=gdal.GDT_Byte) -> None:
    """
    Write 'array' (rows, cols) as GeoTIFF, copying geotransform and projection from ref_path.
    """
    base = gdal.Open(ref_path, gdal.GA_ReadOnly)
    if base is None:
        raise FileNotFoundError(f"Reference not found: {ref_path}")

    rows, cols = array.shape
    driver = gdal.GetDriverByName("GTiff")
    formatOptions = ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]
    out_ds = driver.Create(out_path, cols, rows, 1, dtype, options=formatOptions)

    gt = base.GetGeoTransform()
    proj = base.GetProjectionRef()

    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.FlushCache()
    out_ds = None  # close
    base = None


# =========================
# Chip generation (train)
# =========================
def generate_chips_train(metrics: Dict, labels: Dict, chip: int = GRID) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate non-overlapping chips for training.
    Returns X, Y with shapes:
      X: (N, chip, chip, bands)
      Y: (N, chip, chip, 1)
    """
    rows = metrics["rows"]
    cols = metrics["cols"]
    data_m = metrics["data"]   # (B, rows, cols)
    data_y = labels["data"]    # (1, rows, cols)

    stride = chip  # no overlap for training
    idxs = compute_grid_indices(rows, cols, chip, stride)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for r0, c0 in idxs:
        m_chip = data_m[:, r0:r0 + chip, c0:c0 + chip]               # (B,H,W)
        y_chip = data_y[:, r0:r0 + chip, c0:c0 + chip]               # (1,H,W)
        xs.append(np.transpose(m_chip, (1, 2, 0)))                   # (H,W,B)
        ys.append(np.transpose(y_chip, (1, 2, 0)))                   # (H,W,1)

    X = np.stack(xs).astype(np.float32)
    Y = np.stack(ys).astype(np.float32)
    return X, Y


# =========================
# Offline data augmentation
# =========================
def augment_with_rotations(
    X: np.ndarray,
    Y: np.ndarray,
    k_list: Tuple[int, ...] = (1, 2, 3),
    do_flips: bool = False,
    shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Offline data augmentation for segmentation (X: (N,H,W,C), Y: (N,H,W,1)).
    - k_list: 90°-based rotations to add; 1=90°, 2=180°, 3=270°
    - do_flips: if True, also add horizontal, vertical, and combined flips
    - shuffle: randomly permute the augmented dataset
    WARNING: This multiplies data in memory. Use carefully for large datasets.
    """
    x_parts = [X]
    y_parts = [Y]

    # Rotations by 90° increments (no interpolation artifacts)
    for k in k_list:
        Xr = np.rot90(X, k=k, axes=(1, 2)).copy()
        Yr = np.rot90(Y, k=k, axes=(1, 2)).copy()
        x_parts.append(Xr)
        y_parts.append(Yr)

    # Optional flips
    if do_flips:
        # Horizontal flip
        Xh = X[:, :, ::-1, :].copy()
        Yh = Y[:, :, ::-1, :].copy()
        x_parts.append(Xh)
        y_parts.append(Yh)

        # Vertical flip
        Xv = X[:, ::-1, :, :].copy()
        Yv = Y[:, ::-1, :, :].copy()
        x_parts.append(Xv)
        y_parts.append(Yv)

        # Horizontal + Vertical flip
        Xhv = X[:, ::-1, ::-1, :].copy()
        Yhv = Y[:, ::-1, ::-1, :].copy()
        x_parts.append(Xhv)
        y_parts.append(Yhv)

    X_aug = np.concatenate(x_parts, axis=0)
    Y_aug = np.concatenate(y_parts, axis=0)

    if shuffle:
        idx = np.random.permutation(X_aug.shape[0])
        X_aug = X_aug[idx]
        Y_aug = Y_aug[idx]

    return X_aug, Y_aug


# =========================
# Full-image prediction with blending
# =========================
def predict_full_mosaic(metrics: Dict,
                        model: Model,
                        chip: int = GRID,
                        overlap_ratio: float = 0.5,
                        batch_size: int = BATCH_SIZE_PRED) -> np.ndarray:
    """
    Predict over the full image using sliding windows with overlap and Hann blending.
    Returns an array (rows, cols) in [0,1].
    """
    rows = metrics["rows"]
    cols = metrics["cols"]

    overlap = int(chip * overlap_ratio)
    overlap = min(max(overlap, 0), chip - 1)
    stride = chip - overlap if chip > overlap else 1

    idxs = compute_grid_indices(rows, cols, chip, stride)
    weight_full = np.zeros((rows, cols), dtype=np.float32)
    pred_full = np.zeros((rows, cols), dtype=np.float32)

    w = hann_weight_2d(chip)  # (chip,chip)

    # Batching for efficiency
    batch_X: List[np.ndarray] = []
    batch_pos: List[Tuple[int, int]] = []

    def flush_batch():
        nonlocal batch_X, batch_pos, pred_full, weight_full
        if not batch_X:
            return
        Xb = np.stack(batch_X).astype(np.float32)  # (N, H, W, B)
        P = model.predict_on_batch(Xb)             # (N,H,W,1)
        P = P[..., 0]                              # (N,H,W)

        for k, (r0, c0) in enumerate(batch_pos):
            pred_full[r0:r0 + chip, c0:c0 + chip] += P[k] * w
            weight_full[r0:r0 + chip, c0:c0 + chip] += w

        batch_X = []
        batch_pos = []

    for (r0, c0) in idxs:
        m_chip = metrics["data"][:, r0:r0 + chip, c0:c0 + chip]      # (B,H,W)
        m_chip = np.transpose(m_chip, (1, 2, 0))                     # (H,W,B)
        batch_X.append(m_chip)
        batch_pos.append((r0, c0))

        if len(batch_X) >= batch_size:
            flush_batch()

    flush_batch()

    # Avoid division by zero
    weight_full[weight_full == 0] = 1.0
    mosaic = pred_full / weight_full
    mosaic = np.clip(mosaic, 0.0, 1.0)
    return mosaic


# =========================
# Training
# =========================
def make_train(all_metrics_dir: str,
               tiles_training_list: Union[List[str], str],
               labels_dir: str,
               train_ratio: float = 0.1,
               model_dir: str = "models_256",
               grid: int = GRID,
               n_epochs: int = N_EPOCHS,
               use_rotation_aug: bool = True,
               rotation_k: Tuple[int, ...] = (1, 2, 3),
               use_flip_aug: bool = False) -> str:
    """
    Train the model with the provided tiles and save the best weights.
    New options:
      - use_rotation_aug: apply 90°-based rotation augmentation
      - rotation_k: which rotations to add (1=90°,2=180°,3=270°)
      - use_flip_aug: also add H/V/HV flips
    Returns the path to the best-accuracy weights.
    """
    ensure_dir(model_dir)

    # Support a path to .txt or an in-memory list
    if isinstance(tiles_training_list, str) and os.path.isfile(tiles_training_list):
        with open(tiles_training_list, "r") as f:
            tiles = [line.strip() for line in f if line.strip()]
    else:
        tiles = list(tiles_training_list)

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    for tile_id in tiles:
        # Verify metric files exist
        missing = list_missing(all_metrics_dir, tile_id, METRICS_ORDER)
        if missing is not None:
            print(f"[SKIP] Missing metric '{missing}' for {tile_id}")
            continue

        # Paths
        label_path = os.path.join(labels_dir, f"{tile_id}.tif")
        if not os.path.isfile(label_path):
            print(f"[SKIP] Missing label {label_path}")
            continue

        # Load
        metrics = read_metrics_stack(all_metrics_dir, tile_id, METRICS_ORDER)
        labels = read_label_raster(label_path)

        if metrics["nbands"] != N_METRICS:
            print(f"[SKIP] nbands={metrics['nbands']} != {N_METRICS} in {tile_id}")
            continue

        # Chips (no overlap for training)
        X, Y = generate_chips_train(metrics, labels, chip=grid)
        X_list.append(X)
        Y_list.append(Y)

        # Free
        del X, Y, metrics, labels
        gc.collect()

    if not X_list:
        raise RuntimeError("No valid training data was found.")

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    Y = np.concatenate(Y_list, axis=0).astype(np.float32)
    del X_list, Y_list
    gc.collect()

    # --- Offline augmentation BEFORE the split ---
    if use_rotation_aug or use_flip_aug:
        X, Y = augment_with_rotations(
            X, Y,
            k_list=rotation_k if use_rotation_aug else (),
            do_flips=use_flip_aug,
            shuffle=True
        )

    # Split
    X_tr, X_va, Y_tr, Y_va = train_test_split(
        X, Y, test_size=train_ratio, random_state=SEED, shuffle=True
    )
    del X, Y
    gc.collect()

    # Callbacks
    best_by_acc = os.path.join(model_dir, f"model-{grid}_best_tile_model.h5")
    ckpt_best_acc = kcb.ModelCheckpoint(
        best_by_acc, monitor="val_accuracy", verbose=1,
        save_best_only=True, save_weights_only=True, mode="max"
    )
    ckpt_best_loss = kcb.ModelCheckpoint(
        os.path.join(model_dir, f"model-{grid}_best_loss.h5"),
        monitor="val_loss", verbose=1, save_best_only=True,
        save_weights_only=True, mode="min"
    )
    early = kcb.EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True)
    reduce = kcb.ReduceLROnPlateau(monitor="val_loss", factor=0.8, patience=10, min_lr=1e-5, verbose=1)

    # Model
    model = build_unet(n_metrics=N_METRICS)
    model.fit(
        X_tr, Y_tr,
        validation_data=(X_va, Y_va),
        epochs=n_epochs,
        batch_size=BATCH_SIZE_TRAIN,
        callbacks=[ckpt_best_acc, ckpt_best_loss, early, reduce],
        verbose=1
    )

    # Remove potential intermediate checkpoints
    for f in glob.glob(os.path.join(model_dir, f"model-{grid}_-e*")):
        try:
            os.remove(f)
        except Exception:
            pass

    # Free
    del X_tr, X_va, Y_tr, Y_va, model
    gc.collect()
    tf.keras.backend.clear_session()

    return best_by_acc


# =========================
# Predict tiles
# =========================
def make_predict(all_metrics_dir: str,
                 tiles_predict_list: Union[List[str], str],
                 out_dir: str,
                 model_weights_path: str,
                 year: str,
                 grid: int = GRID,
                 overlap_ratio: float = 0.5) -> None:
    """
    Predict each tile in tiles_predict_list and write GeoTIFF to out_dir/year.
    Uses overlap + Hann blending to reduce chip seam artifacts.
    """
    out_year_dir = os.path.join(out_dir, str(year))
    ensure_dir(out_year_dir)

    # Support a path to .txt or an in-memory list
    if isinstance(tiles_predict_list, str) and os.path.isfile(tiles_predict_list):
        with open(tiles_predict_list, "r") as f:
            tiles = [line.strip() for line in f if line.strip()]
    else:
        tiles = list(tiles_predict_list)

    # Model
    model = build_unet(n_metrics=N_METRICS)
    model.load_weights(model_weights_path)
    print(f"[INFO] Loaded weights: {model_weights_path}")

    missing_log: List[str] = []

    for i, tile_id in enumerate(tiles, 1):
        miss = list_missing(all_metrics_dir, tile_id, METRICS_ORDER)
        if miss is not None:
            print(f"[SKIP] Missing metric '{miss}' for {tile_id}")
            missing_log.append(tile_id)
            continue

        out_path = os.path.join(out_year_dir, f"{tile_id}.tif")
        if os.path.isfile(out_path):
            print(f"[OK] Exists: {out_path}")
            continue

        print(f"[{i}/{len(tiles)}] Predicting: {tile_id}")

        # Load metrics and run full prediction with blending
        metrics = read_metrics_stack(all_metrics_dir, tile_id, METRICS_ORDER)
        if metrics["nbands"] != N_METRICS:
            print(f"[SKIP] nbands={metrics['nbands']} != {N_METRICS} in {tile_id}")
            continue

        mosaic = predict_full_mosaic(metrics, model, chip=grid, overlap_ratio=overlap_ratio, batch_size=BATCH_SIZE_PRED)
        mosaic_u8 = np.clip(np.round(mosaic * OUTPUT_SCALE), 0, 100).astype(np.uint8)

        # Georeference from the first raster in the stack
        ref_path = metrics["ref_path"]
        write_geotiff_like(ref_path, out_path, mosaic_u8, dtype=gdal.GDT_Byte)
        print(f"[OK] Saved: {out_path}")

        # Free per-tile memory
        del metrics, mosaic, mosaic_u8
        gc.collect()

    # Final cleanup of backend
    tf.keras.backend.clear_session()

    # Log tiles with missing metrics
    if missing_log:
        log_path = os.path.join(out_dir, f"tiles_missing_metrics_{year}.txt")
        with open(log_path, "w") as f:
            for t in missing_log:
                f.write(t + "\n")
        print(f"[INFO] Missing-metrics log: {log_path}")
