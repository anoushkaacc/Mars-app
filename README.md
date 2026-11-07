# Mars Terrain Live Inference Dashboard

A one-page, desktop-first Streamlit viewer for continuously cycling through unlabelled Mars test images and displaying selected model prediction overlays.

## Files

- `app.py` — compact live dashboard, prediction caching, checkpoint loading and overlay rendering.
- `models.py` — the five model architectures used by the supported checkpoints.
- `config.py` — checkpoint and test-image paths, class palettes and normalization.
- `requirements.txt` — Python dependencies. Streamlit 1.37+ is required for `st.fragment` timed refresh.
- `run_app.bat` — Windows launcher.

## Folder layout

```text
D:\mars_app\
  app.py
  models.py
  config.py
  requirements.txt
  run_app.bat
  checkpoints\
    unet.pth
    unet_cbam.pth
    CCHNet_FP16.pth
    binary_y_net_mars.pth
    adaptive_y_net_mars.pth
```

Put test images in the folder configured by `TEST_IMAGE_DIR` in `config.py` (default: `D:\Internship\test images`). Ground-truth masks are not required.

## Install and run

```powershell
cd D:\mars_app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The sidebar contains the model selection, overlay opacity, seconds per image and Start/Stop control. While running, the app advances through the test images and refreshes the displayed overlays automatically. The display is frozen when stopped.

Validation mIoU is shown only when a checkpoint contains a `best_miou`, `val_miou` or `miou` metadata value. This is a checkpoint-recorded validation score, not a test-set score. Test-set IoU cannot be calculated from unlabelled images without ground truth.
