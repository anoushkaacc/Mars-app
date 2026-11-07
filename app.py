from __future__ import annotations

import html
import os
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
    FIVE_CLASS_NAMES,
    FIVE_CLASS_PALETTE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TEST_IMAGE_DIR,
    TWELVE_CLASS_NAMES,
    TWELVE_CLASS_PALETTE,
)
from models import build_model, mark_quantizers_initialized


# Public Hugging Face model repository containing the .pth files.
# Change this one value if your repository has a different owner/name.
DEFAULT_HF_REPO_ID = "anoushkaacc/mars-segmentation-checkpoints"
HF_REPO_ID = os.environ.get("HF_REPO_ID", "").strip()
try:
    # Streamlit Community Cloud secrets can override the default repository ID.
    HF_REPO_ID = str(st.secrets.get("HF_REPO_ID", HF_REPO_ID)).strip()
except Exception:
    pass
HF_REPO_ID = HF_REPO_ID or DEFAULT_HF_REPO_ID


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
      html, body, [class*="css"], .stApp { color: var(--ink) !important; }
      .stApp { background: var(--canvas); }
      [data-testid="stHeader"] { background: rgba(247,248,252,.96); }
      [data-testid="stSidebar"] {
        background: var(--paper);
        border-right: 1px solid var(--line);
      }
      [data-testid="stSidebar"] * { color: var(--ink); }
      .block-container {
        max-width: 2400px;
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
        max-height: 65vh;
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
      div[data-testid="stMultiSelect"] [data-baseweb="tag"] * {
        color: #2448bd !important;
      }
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
      .live-pill, .stopped-pill {
        display: inline-block;
        border-radius: 999px;
        padding: .28rem .62rem;
        font-size: .70rem;
        font-weight: 750;
        letter-spacing: .06em;
        white-space: nowrap;
      }
      .live-pill {
        color: #2448bd;
        background: var(--blue-soft);
        border: 1px solid #d8e1ff;
      }
      .stopped-pill {
        color: #8c4564;
        background: var(--pink-soft);
        border: 1px solid #f0d5e1;
      }
      .model-title {
        color: var(--ink);
        font-size: .94rem;
        font-weight: 750;
        line-height: 1.25;
        margin: 0 0 .45rem 0;
        min-height: 1.25rem;
      }
      .image-line {
        color: var(--muted);
        font-size: .78rem;
        margin: .15rem 0 .75rem 0;
      }
      .metric-line {
        display: flex;
        flex-wrap: wrap;
        gap: .35rem .65rem;
        color: #2448bd;
        font-size: .76rem;
        font-weight: 650;
        margin: .42rem 0 .45rem 0;
      }
      .class-legend {
        display: flex;
        flex-wrap: wrap;
        gap: .34rem .55rem;
        margin-top: .35rem;
        padding: .48rem;
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
      }
      .class-item {
        display: inline-flex;
        align-items: center;
        gap: .28rem;
        color: #303746;
        font-size: .70rem;
        line-height: 1.2;
        white-space: nowrap;
      }
      .class-swatch {
        display: inline-block;
        width: 11px;
        height: 11px;
        min-width: 11px;
        border: 1px solid rgba(20, 25, 35, .22);
        border-radius: 2px;
      }
      .class-percent { color: var(--muted); font-variant-numeric: tabular-nums; }
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
        "title": "MarTA-Net (Published IEEE IOT)",
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


def find_local_checkpoint(names: tuple[str, ...]) -> Path | None:
    """Return a locally available checkpoint, if one exists."""
    for name in names:
        for candidate in (CHECKPOINT_DIR / name, APP_DIR / name):
            if candidate.is_file():
                return candidate
    return None


def checkpoint_is_configured(names: tuple[str, ...]) -> bool:
    """Whether a model has a local checkpoint or a configured HF repository."""
    return find_local_checkpoint(names) is not None or bool(HF_REPO_ID)


def get_checkpoint_path(names: tuple[str, ...]) -> str:
    """Use a local checkpoint first; otherwise download it from Hugging Face.

    huggingface_hub is imported lazily so local development can still run with
    local .pth files even before huggingface_hub has been installed.
    """
    local_path = find_local_checkpoint(names)
    if local_path is not None:
        return str(local_path)

    if not HF_REPO_ID:
        raise FileNotFoundError(
            "No local checkpoint was found and HF_REPO_ID is not configured."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download model weights. "
            "Add 'huggingface_hub' to requirements.txt and redeploy."
        ) from exc

    errors: list[str] = []
    for filename in names:
        try:
            # The Hub client caches downloads, so subsequent reruns reuse the
            # cached file rather than downloading the checkpoint again.
            return hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type="model",
            )
        except Exception as exc:
            errors.append(f"{filename}: {type(exc).__name__}: {exc}")

    raise FileNotFoundError(
        f"Could not find any of {', '.join(names)} in Hugging Face repository "
        f"'{HF_REPO_ID}'. Check the repository ID, visibility and exact filenames. "
        + " | ".join(errors)
    )


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
    """Load tensor-only checkpoint data without unrestricted pickle loading."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "PyTorch 2.6 or later is required for safe checkpoint loading."
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
    """Build and load one checkpoint once per Streamlit session resource cache."""
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


def prepare_tensor(
    image_path: Path, size: int, imagenet_normalize: bool
) -> tuple[np.ndarray, torch.Tensor]:
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


def predict_one_image(
    model_key: str,
    checkpoint_path: str,
    image_path: str,
    device_name: str,
) -> tuple[np.ndarray, float]:
    """Run one prediction and return its label map and model-forward latency (ms)."""
    spec = MODEL_SPECS[model_key]
    model, _ = load_model(model_key, checkpoint_path, device_name)
    original_rgb, tensor = prepare_tensor(
        Path(image_path), spec["input_size"], spec["imagenet_normalize"]
    )

    device = torch.device(device_name)
    model_dtype = next(model.parameters()).dtype
    tensor = tensor.to(device=device, dtype=model_dtype)

    # Synchronize CUDA around the forward pass so the timer measures completed
    # GPU work rather than only the asynchronous kernel-launch time.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    with torch.inference_mode():
        output = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_ms = (time.perf_counter() - start_time) * 1000.0

    logits = extract_logits(output)
    if logits.shape[-2:] != (spec["input_size"], spec["input_size"]):
        logits = F.interpolate(
            logits,
            size=(spec["input_size"], spec["input_size"]),
            mode="bilinear",
            align_corners=False,
        )
    labels = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    height, width = original_rgb.shape[:2]
    return resize_labels(labels, width, height), inference_ms


def make_overlay(
    image_rgb: np.ndarray,
    labels: np.ndarray,
    palette: list[tuple[int, int, int]],
    opacity: float,
) -> np.ndarray:
    mask_rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for class_id, rgb in enumerate(palette):
        mask_rgb[labels == class_id] = rgb

    # Keep class 0 (background) visually transparent over the sample image.
    active = labels != 0
    overlay = image_rgb.astype(np.float32).copy()
    overlay[active] = (
        (1.0 - opacity) * overlay[active]
        + opacity * mask_rgb[active].astype(np.float32)
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def classes_for(model_key: str):
    """Return class labels/palette, displaying class 0 as Unknown."""
    if MODEL_SPECS[model_key]["classes"] == 5:
        names = list(FIVE_CLASS_NAMES)
        palette = FIVE_CLASS_PALETTE
    else:
        names = list(TWELVE_CLASS_NAMES)
        palette = TWELVE_CLASS_PALETTE

    if names:
        names[0] = "Unknown"
    return names, palette


def class_legend_html(labels: np.ndarray, model_key: str) -> str:
    """Create a compact color-key legend for classes predicted in this image."""
    names, palette = classes_for(model_key)
    class_ids, pixel_counts = np.unique(labels, return_counts=True)
    total_pixels = max(int(labels.size), 1)
    items: list[str] = []

    for class_id, count in zip(class_ids.tolist(), pixel_counts.tolist()):
        if class_id >= len(names):
            continue
        red, green, blue = palette[class_id]
        name = html.escape(names[class_id])
        percent = (count / total_pixels) * 100.0
        items.append(
            '<span class="class-item">'
            f'<span class="class-swatch" style="background:rgb({red},{green},{blue})"></span>'
            f'<span>{name}</span>'
            f'<span class="class-percent">{percent:.1f}%</span>'
            '</span>'
        )

    if not items:
        return '<div class="class-legend"><span class="class-item">No classes detected</span></div>'
    return '<div class="class-legend">' + "".join(items) + "</div>"


def checkpoint_miou(metadata: dict[str, Any]) -> float | None:
    """Return recorded validation mIoU when checkpoint metadata contains it."""
    for key in ("best_miou", "val_miou", "miou"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            value = float(value)
            return value * 100.0 if 0.0 <= value <= 1.0 else value
    return None


# -----------------------------------------------------------------------------
# Sidebar: compact user controls
# -----------------------------------------------------------------------------
all_keys = list(MODEL_SPECS)
available_keys = [
    key for key in all_keys
    if checkpoint_is_configured(MODEL_SPECS[key]["checkpoint_names"])
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

# On a developer machine, keep using the configured validation/test folder.
# On Streamlit Cloud, the Windows/local path will not exist; use a small folder
# committed with the repository instead.
if TEST_IMAGE_DIR.is_dir():
    ACTIVE_TEST_IMAGE_DIR = TEST_IMAGE_DIR
else:
    ACTIVE_TEST_IMAGE_DIR = APP_DIR / "sample_images"

image_paths = discover_test_images(ACTIVE_TEST_IMAGE_DIR)
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
    f'<div class="sidebar-note">{len(image_paths)} image(s) · {device_name.upper()} · weights: Hugging Face/local</div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Single-page live prediction display
# -----------------------------------------------------------------------------
running = bool(st.session_state.mars_running)
refresh_seconds = float(seconds_per_image) if running else None

st.markdown(
    '<div class="topline"><div><h1>Mars Terrain Inference</h1>'
    '<div class="subline">Sample image and live prediction overlays</div></div>'
    + (
        '<span class="live-pill">LIVE</span>'
        if running
        else '<span class="stopped-pill">STOPPED</span>'
    )
    + '</div>',
    unsafe_allow_html=True,
)

if not ACTIVE_TEST_IMAGE_DIR.is_dir():
    st.info(
        "No image directory was found. Add a 'sample_images' folder containing "
        "at least one JPG or PNG image to your GitHub repository."
    )
elif not image_paths:
    st.info("No supported test images were found in the configured folder.")
elif not available_keys:
    st.info(
        "No models are configured. Check the Hugging Face repository ID at the "
        "top of app.py, or place checkpoint files in the local checkpoints folder."
    )
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
        except Exception as exc:
            st.error(f"Could not open the current image ({type(exc).__name__}).")
            return

        st.markdown(
            f'<div class="image-line">Image {frame_index + 1} of {len(paths)}'
            f' &nbsp;·&nbsp; {html.escape(image_path.name)}</div>',
            unsafe_allow_html=True,
        )

        # First panel is the original image; remaining panels are model overlays.
        columns = st.columns(len(selected_model_keys) + 1, gap="small")
        with columns[0]:
            st.markdown('<div class="model-title">Sample image</div>', unsafe_allow_html=True)
            st.image(image_rgb, use_container_width=True)

        for column, model_key in zip(columns[1:], selected_model_keys):
            spec = MODEL_SPECS[model_key]
            with column:
                st.markdown(
                    f'<div class="model-title">{html.escape(spec["title"])}</div>',
                    unsafe_allow_html=True,
                )
                try:
                    # Resolve/download only selected models, not every model at startup.
                    checkpoint = get_checkpoint_path(spec["checkpoint_names"])
                    _, metadata = load_model(model_key, checkpoint, device_name)
                    labels, inference_ms = predict_one_image(
                        model_key=model_key,
                        checkpoint_path=checkpoint,
                        image_path=str(image_path),
                        device_name=device_name,
                    )
                    overlay = make_overlay(
                        image_rgb,
                        labels,
                        classes_for(model_key)[1],
                        float(opacity),
                    )
                    st.image(overlay, use_container_width=True)

                    metric_parts = [f"Inference · {inference_ms:.2f} ms"]
                    score = checkpoint_miou(metadata)
                    if score is not None:
                        metric_parts.append(f"Validation mIoU · {score:.2f}%")
                    metric_html = "".join(
                        f'<span>{html.escape(part)}</span>' for part in metric_parts
                    )
                    st.markdown(
                        f'<div class="metric-line">{metric_html}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        class_legend_html(labels, model_key),
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    st.warning(
                        f"Prediction unavailable for {spec['title']}: "
                        f"{type(exc).__name__}: {exc}"
                    )

    live_prediction_frame()
