from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    APP_DIR,
    CHECKPOINT_DIR,
    FIVE_CLASS_PALETTE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TEST_IMAGE_DIR,
    TWELVE_CLASS_PALETTE,
)
from models import build_model, mark_quantizers_initialized


# -----------------------------------------------------------------------------
# Page and visual styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Mars Terrain | Live Inference",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --ink: #171923;
        --muted: #626b7a;
        --blue: #315ee8;
        --blue-soft: #edf2ff;
        --pink: #e7a9c0;
        --pink-soft: #fbf0f5;
        --line: #e4e8f0;
        --paper: #ffffff;
        --canvas: #f7f8fc;
      }

      html, body, [class*="css"], .stApp {
        color: var(--ink) !important;
      }
      .stApp { background: var(--canvas); }
      [data-testid="stHeader"] { background: rgba(247,248,252,.96); }
      [data-testid="stSidebar"] {
        background: var(--paper);
        border-right: 1px solid var(--line);
      }
      [data-testid="stSidebar"] * { color: var(--ink); }
      .block-container {
        max-width: 2200px;
        padding-top: 1.05rem;
        padding-bottom: .8rem;
      }
      h1, h2, h3, p, label, span, div { color: inherit; }
      h1 {
        color: var(--ink) !important;
        font-size: 1.55rem !important;
        font-weight: 750 !important;
        letter-spacing: -.035em;
        margin-bottom: .12rem !important;
      }
      [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
      [data-testid="stMarkdownContainer"] p { color: var(--ink); }
      [data-testid="stImage"] img {
        width: 100%;
        max-height: 69vh;
        object-fit: contain;
        border-radius: 8px;
        background: #f0f2f7;
      }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        font-size: 1rem;
        margin-bottom: .55rem;
      }
      div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background: var(--blue-soft);
        border: 1px solid #d8e1ff;
      }
      div[data-testid="stMultiSelect"] [data-baseweb="tag"] * { color: #2448bd !important; }
      div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background: var(--blue);
        border-color: var(--blue);
      }
      div.stButton > button {
        border-radius: 8px;
        border: 1px solid var(--blue);
        color: white !important;
        background: var(--blue);
        font-weight: 650;
        min-height: 2.55rem;
      }
      div.stButton > button:hover {
        border-color: #244bc8;
        background: #244bc8;
        color: white !important;
      }
      div.stButton > button:focus { box-shadow: 0 0 0 .15rem #dce5ff; }
      .topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: .8rem;
      }
      .subline { color: var(--muted); font-size: .82rem; margin-top: .1rem; }
      .live-pill {
        display: inline-block;
        border-radius: 999px;
        padding: .28rem .62rem;
        font-size: .70rem;
        font-weight: 750;
        letter-spacing: .06em;
        color: #2448bd;
        background: var(--blue-soft);
        border: 1px solid #d8e1ff;
        white-space: nowrap;
      }
      .stopped-pill {
        display: inline-block;
        border-radius: 999px;
        padding: .28rem .62rem;
        font-size: .70rem;
        font-weight: 750;
        letter-spacing: .06em;
        color: #8c4564;
        background: var(--pink-soft);
        border: 1px solid #f0d5e1;
        white-space: nowrap;
      }
      .model-title {
        color: var(--ink);
        font-size: .94rem;
        font-weight: 750;
        line-height: 1.25;
        margin: 0 0 .45rem 0;
        min-height: 1.25rem;
      }
      .model-score {
        color: #2448bd;
        font-size: .76rem;
        font-weight: 650;
        margin-top: .35rem;
      }
      .model-no-score {
        color: var(--muted);
        font-size: .72rem;
        margin-top: .35rem;
      }
      .image-line {
        color: var(--muted);
        font-size: .78rem;
        margin: .15rem 0 .75rem 0;
      }
      .sidebar-note {
        color: var(--muted);
        font-size: .75rem;
        line-height: 1.4;
      }
      [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] {
        gap: .35rem;
      }
      hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model/checkpoint configuration
# -----------------------------------------------------------------------------
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "unet": {
        "title": "U-Net",
        "classes": 5,
        "checkpoint_names": ("unet.pth",),
        "input_size": 256,
        "imagenet_normalize": False,
    },
    "unet_cbam": {
        "title": "U-Net + CBAM",
        "classes": 5,
        "checkpoint_names": ("unet_cbam.pth",),
        "input_size": 256,
        "imagenet_normalize": False,
    },
    "cchnet_fp16": {
        "title": "CCHNet · FP16",
        "classes": 12,
        "checkpoint_names": ("CCHNet_FP16.pth", "cchnet_fp16.pth"),
        "input_size": 384,
        "imagenet_normalize": True,
    },
    "cchnet_binary": {
        "title": "CCHNet · Binarized",
        "classes": 12,
        "checkpoint_names": ("binary_y_net_mars.pth",),
        "input_size": 384,
        "imagenet_normalize": True,
    },
    "cchnet_quant": {
        "title": "CCHNet · 4-bit LSQ",
        "classes": 12,
        "checkpoint_names": ("adaptive_y_net_mars.pth",),
        "input_size": 384,
        "imagenet_normalize": True,
    },
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def resolve_checkpoint(names: tuple[str, ...]) -> Path | None:
    """Resolve a checkpoint from the configured folder, then project root."""
    for name in names:
        for candidate in (CHECKPOINT_DIR / name, APP_DIR / name):
            if candidate.is_file():
                return candidate
    return None


def discover_test_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        return []
    return sorted(
        (
            path for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: (path.parent.as_posix().lower(), path.name.lower()),
    )


def read_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Read tensor-only checkpoint data without unrestricted pickle loading."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Install PyTorch 2.6 or later; this app requires weights_only checkpoint loading."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Unsupported checkpoint format.")

    metadata = {
        key: payload[key]
        for key in ("epoch", "best_miou", "val_miou", "miou", "quant_bits")
        if key in payload and isinstance(payload[key], (int, float))
    }

    if "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif "state_dict" in payload:
        state = payload["state_dict"]
    elif payload and all(torch.is_tensor(value) for value in payload.values()):
        state = payload
        metadata = {}
    else:
        raise ValueError("No supported model state dictionary was found.")

    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint contains an empty state dictionary.")

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if not torch.is_tensor(value):
            raise ValueError(f"Non-tensor checkpoint entry: {key}")
        cleaned[clean_key] = value
    return cleaned, metadata


@st.cache_resource(show_spinner=False)
def load_model(model_key: str, checkpoint_path: str, device_name: str):
    device = torch.device(device_name)
    model = build_model(model_key)
    state_dict, metadata = read_checkpoint(Path(checkpoint_path))
    model.load_state_dict(state_dict, strict=True)

    if model_key == "cchnet_quant":
        mark_quantizers_initialized(model)

    model.to(device)
    if model_key == "cchnet_fp16" and device.type == "cuda":
        model.half()
    model.eval()
    return model, metadata


def prepare_tensor(image_path: Path, size: int, imagenet_normalize: bool) -> tuple[np.ndarray, torch.Tensor]:
    with Image.open(image_path) as source:
        original_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        resized = source.convert("RGB").resize(
            (size, size), resample=Image.Resampling.BILINEAR
        )

    array = np.asarray(resized, dtype=np.uint8)
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0
    if imagenet_normalize:
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean) / std
    return original_rgb, tensor.unsqueeze(0)


def extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("Model returned no output tensors.")
        output = output[0]
    elif isinstance(output, dict):
        for key in ("terrain_logits", "logits", "out"):
            if key in output:
                output = output[key]
                break
        else:
            raise RuntimeError("Model output did not contain segmentation logits.")

    if not torch.is_tensor(output):
        raise RuntimeError("Model output is not a tensor.")
    return output


def resize_labels(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    label_image = Image.fromarray(labels.astype(np.uint8), mode="L")
    return np.asarray(
        label_image.resize((width, height), resample=Image.Resampling.NEAREST),
        dtype=np.uint8,
    )


@st.cache_data(show_spinner=False, max_entries=256)
def predict_one_image(
    model_key: str,
    checkpoint_path: str,
    checkpoint_mtime_ns: int,
    image_path: str,
    image_mtime_ns: int,
    device_name: str,
) -> np.ndarray:
    """Cache one image/model prediction; mtime keys refresh changed files."""
    del checkpoint_mtime_ns, image_mtime_ns  # Used as cache-key version values.
    spec = MODEL_SPECS[model_key]
    model, _ = load_model(model_key, checkpoint_path, device_name)
    original_rgb, tensor = prepare_tensor(
        Path(image_path), spec["input_size"], spec["imagenet_normalize"]
    )
    device = torch.device(device_name)
    model_dtype = next(model.parameters()).dtype
    tensor = tensor.to(device=device, dtype=model_dtype)

    with torch.inference_mode():
        logits = extract_logits(model(tensor))
        if logits.shape[-2:] != (spec["input_size"], spec["input_size"]):
            logits = F.interpolate(
                logits,
                size=(spec["input_size"], spec["input_size"]),
                mode="bilinear",
                align_corners=False,
            )
        labels = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    height, width = original_rgb.shape[:2]
    return resize_labels(labels, width, height)


def make_overlay(
    image_rgb: np.ndarray,
    labels: np.ndarray,
    palette: list[tuple[int, int, int]],
    opacity: float,
) -> np.ndarray:
    mask_rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for class_id, rgb in enumerate(palette):
        mask_rgb[labels == class_id] = rgb

    # Preserve the source image for class zero (background).
    active = labels != 0
    overlay = image_rgb.astype(np.float32).copy()
    overlay[active] = (
        (1.0 - opacity) * overlay[active]
        + opacity * mask_rgb[active].astype(np.float32)
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def palette_for(model_key: str) -> list[tuple[int, int, int]]:
    return FIVE_CLASS_PALETTE if MODEL_SPECS[model_key]["classes"] == 5 else TWELVE_CLASS_PALETTE


def checkpoint_miou(metadata: dict[str, Any]) -> float | None:
    """Return checkpoint-recorded validation mIoU, if included in metadata."""
    for key in ("best_miou", "val_miou", "miou"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    return None


# -----------------------------------------------------------------------------
# Sidebar: the only user-adjustable controls
# -----------------------------------------------------------------------------
all_keys = list(MODEL_SPECS)
available_keys = [
    key for key in all_keys
    if resolve_checkpoint(MODEL_SPECS[key]["checkpoint_names"]) is not None
]

st.sidebar.markdown("## Display controls")
selected_model_keys = st.sidebar.multiselect(
    "Models",
    options=available_keys,
    default=available_keys,
    format_func=lambda key: MODEL_SPECS[key]["title"],
    placeholder="Select models",
)
opacity = st.sidebar.slider(
    "Overlay opacity",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)
seconds_per_image = st.sidebar.number_input(
    "Seconds per image",
    min_value=1.0,
    max_value=60.0,
    value=3.0,
    step=1.0,
)

image_paths = discover_test_images(TEST_IMAGE_DIR)
device_name = "cuda" if torch.cuda.is_available() else "cpu"

if "mars_running" not in st.session_state:
    st.session_state.mars_running = False
if "mars_image_index" not in st.session_state:
    st.session_state.mars_image_index = 0
if "mars_last_image_index" not in st.session_state:
    st.session_state.mars_last_image_index = 0

if not selected_model_keys or not image_paths:
    st.session_state.mars_running = False

button_label = "Stop" if st.session_state.mars_running else "Start"
if st.sidebar.button(
    button_label,
    type="primary",
    use_container_width=True,
    disabled=(not selected_model_keys or not image_paths),
):
    if st.session_state.mars_running:
        st.session_state.mars_running = False
    else:
        st.session_state.mars_running = True
        st.session_state.mars_image_index = 0
        st.session_state.mars_last_image_index = 0
    st.rerun()

st.sidebar.markdown(
    f'<div class="sidebar-note">{len(image_paths)} test image(s) · {device_name.upper()}</div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Single-page live prediction display
# -----------------------------------------------------------------------------
running = bool(st.session_state.mars_running)
refresh_seconds = float(seconds_per_image) if running else None

st.markdown(
    '<div class="topline"><div><h1>Mars Terrain Inference</h1>'
    '<div class="subline">Live prediction overlays</div></div>'
    + (
        '<span class="live-pill">LIVE</span>'
        if running
        else '<span class="stopped-pill">STOPPED</span>'
    )
    + '</div>',
    unsafe_allow_html=True,
)

if not TEST_IMAGE_DIR.is_dir():
    st.info("Test image folder not found. Check TEST_IMAGE_DIR in config.py.")
elif not image_paths:
    st.info("No supported test images were found in the configured folder.")
elif not available_keys:
    st.info("No supported model checkpoints were found in the checkpoints folder.")
elif not selected_model_keys:
    st.info("Select at least one model in the sidebar.")
else:
    @st.fragment(run_every=refresh_seconds)
    def live_prediction_frame():
        is_running = bool(st.session_state.mars_running)
        paths = image_paths

        if is_running:
            frame_index = int(st.session_state.mars_image_index) % len(paths)
            st.session_state.mars_last_image_index = frame_index
            st.session_state.mars_image_index = (frame_index + 1) % len(paths)
        else:
            frame_index = int(st.session_state.mars_last_image_index) % len(paths)

        image_path = paths[frame_index]
        try:
            with Image.open(image_path) as source:
                image_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        except Exception:
            st.error("The current image could not be opened.")
            return

        st.markdown(
            f'<div class="image-line">Image {frame_index + 1} of {len(paths)}'
            f' &nbsp;·&nbsp; {image_path.name}</div>',
            unsafe_allow_html=True,
        )

        columns = st.columns(len(selected_model_keys), gap="small")
        for column, model_key in zip(columns, selected_model_keys):
            spec = MODEL_SPECS[model_key]
            checkpoint = resolve_checkpoint(spec["checkpoint_names"])
            with column:
                st.markdown(
                    f'<div class="model-title">{spec["title"]}</div>',
                    unsafe_allow_html=True,
                )
                if checkpoint is None:
                    st.caption("Checkpoint unavailable")
                    continue

                try:
                    _, metadata = load_model(model_key, str(checkpoint), device_name)
                    labels = predict_one_image(
                        model_key=model_key,
                        checkpoint_path=str(checkpoint),
                        checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
                        image_path=str(image_path),
                        image_mtime_ns=image_path.stat().st_mtime_ns,
                        device_name=device_name,
                    )
                    overlay = make_overlay(
                        image_rgb,
                        labels,
                        palette_for(model_key),
                        float(opacity),
                    )
                    st.image(overlay, use_container_width=True)

                    score = checkpoint_miou(metadata)
                    if score is not None:
                        st.markdown(
                            f'<div class="model-score">Validation mIoU · {score * 100:.2f}%</div>',
                            unsafe_allow_html=True,
                        )
                except Exception:
                    st.warning("Prediction unavailable for this model.")

    live_prediction_frame()
