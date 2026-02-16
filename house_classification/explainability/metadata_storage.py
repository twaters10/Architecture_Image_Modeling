#!/usr/bin/env python3
"""
Efficient Storage of Explainability Metadata.

Instead of saving thousands of full-resolution heatmap images, store compact
metadata that captures the essential information for later analysis and debugging.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime


@dataclass
class ExplainabilityMetadata:
    """
    Compact metadata for a single Grad-CAM explanation.

    Stores essential information without full heatmap images, enabling
    efficient storage and fast querying for production pipelines.
    """
    # Image identification
    image_path: str
    image_id: Optional[str] = None
    timestamp: Optional[str] = None

    # Model information
    model_name: str = ""
    checkpoint_path: str = ""

    # Prediction results
    predicted_class: str = ""
    predicted_class_idx: int = -1
    confidence: float = 0.0
    true_class: Optional[str] = None
    is_correct: Optional[bool] = None

    # Top-K predictions
    top_k_classes: List[str] = None
    top_k_confidences: List[float] = None

    # Heatmap statistics (instead of full heatmap)
    heatmap_mean: float = 0.0
    heatmap_std: float = 0.0
    heatmap_max: float = 0.0
    heatmap_min: float = 0.0
    heatmap_percentile_95: float = 0.0
    heatmap_percentile_50: float = 0.0

    # Spatial distribution (which quadrants have high activation)
    top_left_activation: float = 0.0
    top_right_activation: float = 0.0
    bottom_left_activation: float = 0.0
    bottom_right_activation: float = 0.0
    center_activation: float = 0.0

    # Sanity metrics
    confidence_drop_pct: float = 0.0
    confidence_increase_pct: float = 0.0
    concentration: float = 0.0
    coverage: float = 0.0
    is_suspicious: bool = False
    suspicious_reason: str = ""

    # Optional: compressed heatmap representation
    # Store top-N (x, y, intensity) points instead of full array
    top_activations: Optional[List[Dict[str, float]]] = None

    def __post_init__(self):
        """Initialize optional fields."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.top_k_classes is None:
            self.top_k_classes = []
        if self.top_k_confidences is None:
            self.top_k_confidences = []
        if self.is_correct is None and self.true_class is not None:
            self.is_correct = (self.predicted_class == self.true_class)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExplainabilityMetadata':
        """Create from dictionary."""
        return cls(**data)


class MetadataStorage:
    """
    Efficient storage backend for explainability metadata.

    Supports both JSON (human-readable, good for small datasets) and
    JSONL (line-delimited, good for streaming large datasets).
    """

    def __init__(self, output_path: Path, format: str = "jsonl"):
        """
        Initialize metadata storage.

        Args:
            output_path: Path to output file (.json or .jsonl).
            format: Storage format - "json" or "jsonl" (default: "jsonl").
        """
        self.output_path = Path(output_path)
        self.format = format
        self.metadata_list: List[ExplainabilityMetadata] = []

        # Ensure parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, metadata: ExplainabilityMetadata):
        """Add metadata for a single explanation."""
        self.metadata_list.append(metadata)

    def save(self):
        """Save all metadata to disk."""
        if self.format == "jsonl":
            self._save_jsonl()
        else:
            self._save_json()

    def _save_json(self):
        """Save as single JSON array (good for small datasets)."""
        data = [m.to_dict() for m in self.metadata_list]
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} metadata records to: {self.output_path}")

    def _save_jsonl(self):
        """Save as line-delimited JSON (good for streaming large datasets)."""
        with open(self.output_path, 'w') as f:
            for metadata in self.metadata_list:
                json.dump(metadata.to_dict(), f)
                f.write('\n')
        print(f"Saved {len(self.metadata_list)} metadata records to: {self.output_path}")

    def append_to_file(self, metadata: ExplainabilityMetadata):
        """
        Append single record to file immediately (streaming mode).

        Useful for production pipelines processing large batches - avoids
        holding all metadata in memory.
        """
        if self.format != "jsonl":
            raise ValueError("Append mode only supported for jsonl format")

        with open(self.output_path, 'a') as f:
            json.dump(metadata.to_dict(), f)
            f.write('\n')

    @classmethod
    def load(cls, file_path: Path) -> List[ExplainabilityMetadata]:
        """
        Load metadata from file.

        Args:
            file_path: Path to .json or .jsonl file.

        Returns:
            List of ExplainabilityMetadata objects.
        """
        file_path = Path(file_path)

        if file_path.suffix == '.jsonl':
            return cls._load_jsonl(file_path)
        else:
            return cls._load_json(file_path)

    @classmethod
    def _load_json(cls, file_path: Path) -> List[ExplainabilityMetadata]:
        """Load from JSON array."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return [ExplainabilityMetadata.from_dict(d) for d in data]

    @classmethod
    def _load_jsonl(cls, file_path: Path) -> List[ExplainabilityMetadata]:
        """Load from line-delimited JSON."""
        metadata_list = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    metadata_list.append(ExplainabilityMetadata.from_dict(json.loads(line)))
        return metadata_list

    @staticmethod
    def query_suspicious(metadata_list: List[ExplainabilityMetadata]) -> List[ExplainabilityMetadata]:
        """Filter for suspicious predictions flagged by sanity metrics."""
        return [m for m in metadata_list if m.is_suspicious]

    @staticmethod
    def query_misclassifications(metadata_list: List[ExplainabilityMetadata]) -> List[ExplainabilityMetadata]:
        """Filter for misclassified predictions."""
        return [m for m in metadata_list if m.is_correct is False]

    @staticmethod
    def query_by_class(metadata_list: List[ExplainabilityMetadata], class_name: str) -> List[ExplainabilityMetadata]:
        """Filter by predicted class."""
        return [m for m in metadata_list if m.predicted_class == class_name]

    @staticmethod
    def query_low_confidence(
        metadata_list: List[ExplainabilityMetadata],
        threshold: float = 0.5
    ) -> List[ExplainabilityMetadata]:
        """Filter for low-confidence predictions."""
        return [m for m in metadata_list if m.confidence < threshold]


def extract_heatmap_statistics(heatmap: np.ndarray) -> Dict[str, float]:
    """
    Extract statistical summary of heatmap (instead of storing full array).

    Args:
        heatmap: Grad-CAM heatmap (H, W) in [0, 1].

    Returns:
        Dictionary of statistics.
    """
    return {
        'heatmap_mean': float(np.mean(heatmap)),
        'heatmap_std': float(np.std(heatmap)),
        'heatmap_max': float(np.max(heatmap)),
        'heatmap_min': float(np.min(heatmap)),
        'heatmap_percentile_95': float(np.percentile(heatmap, 95)),
        'heatmap_percentile_50': float(np.percentile(heatmap, 50)),
    }


def extract_spatial_distribution(heatmap: np.ndarray) -> Dict[str, float]:
    """
    Extract spatial distribution of activations (which regions are active).

    Divides heatmap into quadrants and center, computes mean activation for each.

    Args:
        heatmap: Grad-CAM heatmap (H, W) in [0, 1].

    Returns:
        Dictionary of spatial statistics.
    """
    h, w = heatmap.shape
    h_mid, w_mid = h // 2, w // 2

    # Quadrants
    top_left = heatmap[:h_mid, :w_mid]
    top_right = heatmap[:h_mid, w_mid:]
    bottom_left = heatmap[h_mid:, :w_mid]
    bottom_right = heatmap[h_mid:, w_mid:]

    # Center (middle 50%)
    h_quarter, w_quarter = h // 4, w // 4
    center = heatmap[h_quarter:3*h_quarter, w_quarter:3*w_quarter]

    return {
        'top_left_activation': float(np.mean(top_left)),
        'top_right_activation': float(np.mean(top_right)),
        'bottom_left_activation': float(np.mean(bottom_left)),
        'bottom_right_activation': float(np.mean(bottom_right)),
        'center_activation': float(np.mean(center)),
    }


def extract_top_activations(heatmap: np.ndarray, top_n: int = 10) -> List[Dict[str, float]]:
    """
    Extract top-N highest activation points (compact representation).

    Instead of storing full heatmap, store coordinates and intensities of
    top activations. Useful for approximate reconstruction or debugging.

    Args:
        heatmap: Grad-CAM heatmap (H, W) in [0, 1].
        top_n: Number of top points to extract.

    Returns:
        List of {x, y, intensity} dicts.
    """
    h, w = heatmap.shape
    flat_indices = np.argsort(heatmap.ravel())[::-1][:top_n]

    top_points = []
    for idx in flat_indices:
        y, x = divmod(int(idx), w)
        intensity = float(heatmap[y, x])
        # Normalize coordinates to [0, 1] for resolution-independence
        top_points.append({
            'x': x / w,
            'y': y / h,
            'intensity': intensity
        })

    return top_points
