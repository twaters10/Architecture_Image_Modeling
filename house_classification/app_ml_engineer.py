#!/usr/bin/env python3
"""
Streamlit Web App for Architectural Style Classification.

Upload an image and get predictions with Grad-CAM visual explanations.

Usage:
    streamlit run app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from inference import InferencePipeline
from utils.config import get_data_paths


def find_checkpoint_dirs() -> list[Path]:
    """Scan checkpoints/ for directories containing best_model.pth."""
    paths = get_data_paths()
    checkpoints_root = paths["checkpoints"]
    if not checkpoints_root.exists():
        return []
    dirs = []
    for d in sorted(checkpoints_root.iterdir()):
        if d.is_dir() and (d / "best_model.pth").exists():
            dirs.append(d)
    return dirs


@st.cache_resource
def load_pipeline(checkpoint_path: str) -> InferencePipeline:
    """Load and cache the InferencePipeline so it persists across reruns."""
    return InferencePipeline(checkpoint_path)


def display_name(class_name: str) -> str:
    """Convert snake_case class name to Title Case (e.g. tudor_revival -> Tudor Revival)."""
    return class_name.replace("_", " ").title()


def main():
    st.set_page_config(
        page_title="Architectural Style Classifier",
        page_icon="🏛",
        layout="wide",
    )

    st.title("Architectural Style Classifier")
    st.caption("Upload an image to classify its architectural style with Grad-CAM explanations.")

    # ── Sidebar ──────────────────────────────────────────────────────────
    st.sidebar.header("Model Settings")

    checkpoint_dirs = find_checkpoint_dirs()
    if not checkpoint_dirs:
        st.sidebar.error("No checkpoints found in checkpoints/ directory. Train a model first.")
        st.stop()

    selected_dir = st.sidebar.selectbox(
        "Checkpoint folder",
        checkpoint_dirs,
        format_func=lambda d: d.name,
    )

    checkpoint_path = str(selected_dir / "best_model.pth")

    top_k = st.sidebar.slider("Top-K predictions", min_value=1, max_value=10, value=5)

    # ── Main area ────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an image to get started.")
        st.stop()

    image = Image.open(uploaded).convert("RGB")

    # Load pipeline (cached)
    try:
        pipe = load_pipeline(checkpoint_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Run inference
    with st.spinner("Classifying..."):
        result = pipe.explain(image, top_k=top_k)

    # ── Row 1: Metrics ───────────────────────────────────────────────────
    col_pred, col_conf = st.columns(2)
    col_pred.metric("Predicted Class", display_name(result["predicted_class"]))
    col_conf.metric("Confidence", f"{result['confidence']:.1%}")

    # ── Row 2: Top-K bar chart ───────────────────────────────────────────
    st.subheader(f"Top-{top_k} Predictions")
    raw_names = [name for name, _ in result["top_k"]]
    names = [display_name(n) for n in raw_names]
    probs = [prob for _, prob in result["top_k"]]

    fig, ax = plt.subplots(figsize=(8, max(2, 0.5 * top_k)))
    y_pos = list(range(len(names)))
    colors = ["#2ecc71" if n == result["predicted_class"] else "#3498db" for n in raw_names]
    ax.barh(y_pos, probs, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.invert_yaxis()
    for i, p in enumerate(probs):
        ax.text(p + 0.01, i, f"{p:.1%}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Row 3: Grad-CAM visualizations ───────────────────────────────────
    st.subheader("Grad-CAM Explanation")
    col_orig, col_over = st.columns(2)

    with col_orig:
        st.image(image, caption="Original", use_container_width=True)

    with col_over:
        overlay_uint8 = (result["overlay"] * 255).astype(np.uint8)
        st.image(overlay_uint8, caption="Overlay", use_container_width=True)


if __name__ == "__main__":
    main()
