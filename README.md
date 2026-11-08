# Martian Terrain Segmentation Dashboard

**VG-NET Lab, IISER Thiruvananthapuram**\
**Research support: Indian Space Research Organisation (ISRO)**

An interactive Streamlit application for visualizing and comparing
deep-learning models for multi-class semantic segmentation of Martian
surface imagery.

## Live Demo

**[Explore the Mars Segmentation
Dashboard](https://mars-app-inference-only.streamlit.app/)**

Try the application to compare model predictions on Martian terrain
images.

## What does the application do?

-   **Model comparison:** Visualize predictions from multiple
    segmentation architectures.
-   **Prediction overlays:** Compare predicted terrain masks against the
    original sample image.
-   **Class-wise analysis:** Identify terrain classes present in each
    prediction, with color-coded labels and pixel percentages.
-   **Inference timing:** View the time taken by each model to generate
    a prediction.
-   **Model evaluation:** Display available validation mIoU information
    when checkpoint metadata is available.
-   **Interactive exploration:** Select models, adjust overlay opacity,
    and cycle through sample images.

## Models

  Model                               | Status
  ------------------------------------ --------------
  **MarTA-Net (Published IEEE IoT)**  | Published
  U-Net                               | Baseline
  CCHNet -- Binarized                 | Experimental
  CCHNet -- 4-bit LSQ                 | Experimental

## Featured Research: MarTA-Net

MarTA-Net is a terrain-aware U-Net enhanced with the **Convolutional
Block Attention Module (CBAM)** for multi-class Martian surface
segmentation. The published work investigates attention-based feature
refinement and a lightweight model direction for more efficient
inference.

**Interested in the methodology and experimental results?**

Read the paper:

-   **Publication:** [MarTA-Net: A Terrain-Aware U-Net Enhanced With
    Convolutional Block Attention Modules for Martian Surface
    Segmentation](https://doi.org/10.1109/JIOT.2026.3726564)
-   **Journal:** *IEEE Internet of Things Journal*, 2026
-   **Dataset:** [MarTA-Net Dataset on Hugging
    Face](https://huggingface.co/datasets/Anoushkaacc/MarTA-Net)

## Acknowledgement

This research is conducted at **VG-NET Lab, IISER Thiruvananthapuram**,
with support from the **Indian Space Research Organisation (ISRO)**.
