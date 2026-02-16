#!/usr/bin/env python3
"""
Automated Sanity Metrics for Grad-CAM Heatmaps.

Provides quantitative metrics to programmatically assess the quality and
reliability of Grad-CAM explanations for production inference pipelines.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMMetrics:
    """
    Compute automated sanity metrics for Grad-CAM heatmaps.

    These metrics help identify suspicious predictions or low-quality explanations
    that may require human review in production systems.
    """

    @staticmethod
    def compute_confidence_drop(
        model: nn.Module,
        input_tensor: torch.Tensor,
        heatmap: np.ndarray,
        target_class: int,
        mask_threshold: float = 0.5,
        device: Optional[torch.device] = None
    ) -> float:
        """
        Measure drop in confidence when masking high-activation regions.

        A good explanation should show significant confidence drop when important
        regions are masked. Low drop suggests the model isn't actually using those
        regions for prediction.

        Args:
            model: Trained model.
            input_tensor: Original input tensor (1, C, H, W).
            heatmap: Grad-CAM heatmap (H, W) in [0, 1].
            target_class: Target class index.
            mask_threshold: Threshold for considering regions as "important" (default: 0.5).
            device: Device to run on.

        Returns:
            Confidence drop percentage (0-100). Higher is better.

        Example:
            >>> drop = GradCAMMetrics.compute_confidence_drop(model, input_tensor, heatmap, pred_class)
            >>> if drop < 10:
            >>>     print("Warning: Low confidence drop - explanation may be unreliable")
        """
        if device is None:
            device = input_tensor.device

        model.eval()

        # Get original confidence
        with torch.no_grad():
            original_output = model(input_tensor)
            original_prob = F.softmax(original_output, dim=1)[0, target_class].item()

        # Resize heatmap to match input spatial dimensions
        _, _, h, w = input_tensor.shape
        heatmap_resized = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
        heatmap_resized = F.interpolate(heatmap_resized, size=(h, w), mode='bilinear', align_corners=False)
        heatmap_resized = heatmap_resized.squeeze().to(device)

        # Create binary mask for high-activation regions
        mask = (heatmap_resized >= mask_threshold).float()

        # Apply mask (set high-activation regions to mean value)
        masked_input = input_tensor.clone()
        for c in range(masked_input.shape[1]):  # For each channel
            masked_input[0, c] = masked_input[0, c] * (1 - mask)

        # Get masked confidence
        with torch.no_grad():
            masked_output = model(masked_input)
            masked_prob = F.softmax(masked_output, dim=1)[0, target_class].item()

        # Compute drop percentage
        drop_pct = (original_prob - masked_prob) / original_prob * 100 if original_prob > 0 else 0
        return max(0, drop_pct)  # Clamp to non-negative

    @staticmethod
    def compute_confidence_increase(
        model: nn.Module,
        input_tensor: torch.Tensor,
        heatmap: np.ndarray,
        target_class: int,
        mask_threshold: float = 0.3,
        device: Optional[torch.device] = None
    ) -> float:
        """
        Measure increase in confidence when masking low-activation (irrelevant) regions.

        Masking irrelevant regions should either maintain or increase confidence,
        as noise is removed. Decrease suggests the heatmap missed important features.

        Args:
            model: Trained model.
            input_tensor: Original input tensor (1, C, H, W).
            heatmap: Grad-CAM heatmap (H, W) in [0, 1].
            target_class: Target class index.
            mask_threshold: Threshold for considering regions as "irrelevant" (default: 0.3).
            device: Device to run on.

        Returns:
            Confidence change percentage (-100 to +100). Positive is better.

        Example:
            >>> increase = GradCAMMetrics.compute_confidence_increase(model, input_tensor, heatmap, pred_class)
            >>> if increase < -20:
            >>>     print("Warning: Masking irrelevant regions decreased confidence - possible missed features")
        """
        if device is None:
            device = input_tensor.device

        model.eval()

        # Get original confidence
        with torch.no_grad():
            original_output = model(input_tensor)
            original_prob = F.softmax(original_output, dim=1)[0, target_class].item()

        # Resize heatmap to match input spatial dimensions
        _, _, h, w = input_tensor.shape
        heatmap_resized = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
        heatmap_resized = F.interpolate(heatmap_resized, size=(h, w), mode='bilinear', align_corners=False)
        heatmap_resized = heatmap_resized.squeeze().to(device)

        # Create binary mask for low-activation (irrelevant) regions
        mask = (heatmap_resized < mask_threshold).float()

        # Apply mask (set low-activation regions to mean value)
        masked_input = input_tensor.clone()
        for c in range(masked_input.shape[1]):
            masked_input[0, c] = masked_input[0, c] * (1 - mask)

        # Get masked confidence
        with torch.no_grad():
            masked_output = model(masked_input)
            masked_prob = F.softmax(masked_output, dim=1)[0, target_class].item()

        # Compute change percentage
        change_pct = (masked_prob - original_prob) / original_prob * 100 if original_prob > 0 else 0
        return change_pct

    @staticmethod
    def compute_heatmap_concentration(heatmap: np.ndarray) -> float:
        """
        Measure how concentrated the heatmap is (vs. diffuse/uniform).

        High concentration means attention is focused on specific regions (good).
        Low concentration suggests the model is uncertain or the explanation is poor.

        Uses entropy-based metric: lower entropy = more concentrated.

        Args:
            heatmap: Grad-CAM heatmap (H, W) in [0, 1].

        Returns:
            Concentration score (0-1). Higher = more concentrated (better).

        Example:
            >>> concentration = GradCAMMetrics.compute_heatmap_concentration(heatmap)
            >>> if concentration < 0.3:
            >>>     print("Warning: Diffuse heatmap - model may be uncertain")
        """
        # Flatten and normalize to probability distribution
        heatmap_flat = heatmap.flatten()
        heatmap_flat = heatmap_flat / (heatmap_flat.sum() + 1e-10)

        # Compute entropy
        epsilon = 1e-10
        entropy = -np.sum(heatmap_flat * np.log(heatmap_flat + epsilon))

        # Normalize by max possible entropy (uniform distribution)
        max_entropy = np.log(len(heatmap_flat))

        # Convert to concentration (1 - normalized_entropy)
        concentration = 1 - (entropy / max_entropy)
        return float(concentration)

    @staticmethod
    def compute_coverage(heatmap: np.ndarray, threshold: float = 0.2) -> float:
        """
        Measure the spatial coverage of the heatmap.

        Coverage indicates what fraction of the image the model considers relevant.
        Very low coverage might indicate overfitting to small features.
        Very high coverage might indicate poor discrimination.

        Args:
            heatmap: Grad-CAM heatmap (H, W) in [0, 1].
            threshold: Minimum activation to consider a region "active".

        Returns:
            Coverage ratio (0-1). Typical good range: 0.1-0.4.

        Example:
            >>> coverage = GradCAMMetrics.compute_coverage(heatmap)
            >>> if coverage > 0.7:
            >>>     print("Warning: Very high coverage - model may not be discriminative")
        """
        active_pixels = (heatmap >= threshold).sum()
        total_pixels = heatmap.size
        return float(active_pixels / total_pixels)

    @staticmethod
    def compute_all_metrics(
        model: nn.Module,
        input_tensor: torch.Tensor,
        heatmap: np.ndarray,
        target_class: int,
        device: Optional[torch.device] = None
    ) -> Dict[str, float]:
        """
        Compute all sanity metrics for a Grad-CAM heatmap.

        Args:
            model: Trained model.
            input_tensor: Original input tensor (1, C, H, W).
            heatmap: Grad-CAM heatmap (H, W) in [0, 1].
            target_class: Target class index.
            device: Device to run on.

        Returns:
            Dictionary of all metrics.

        Example:
            >>> metrics = GradCAMMetrics.compute_all_metrics(model, input_tensor, heatmap, pred_class)
            >>> if metrics['confidence_drop'] < 10:
            >>>     print("Warning: Low confidence drop")
            >>> if metrics['concentration'] < 0.3:
            >>>     print("Warning: Diffuse heatmap")
        """
        return {
            'confidence_drop_pct': GradCAMMetrics.compute_confidence_drop(
                model, input_tensor, heatmap, target_class, device=device
            ),
            'confidence_increase_pct': GradCAMMetrics.compute_confidence_increase(
                model, input_tensor, heatmap, target_class, device=device
            ),
            'concentration': GradCAMMetrics.compute_heatmap_concentration(heatmap),
            'coverage': GradCAMMetrics.compute_coverage(heatmap)
        }

    @staticmethod
    def flag_suspicious_prediction(metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Flag predictions that may need human review based on metric thresholds.

        Args:
            metrics: Dictionary of metrics from compute_all_metrics().

        Returns:
            Tuple of (is_suspicious, reason).

        Example:
            >>> metrics = GradCAMMetrics.compute_all_metrics(model, input_tensor, heatmap, pred_class)
            >>> is_suspicious, reason = GradCAMMetrics.flag_suspicious_prediction(metrics)
            >>> if is_suspicious:
            >>>     print(f"FLAGGED: {reason}")
            >>>     # Send to human review queue
        """
        reasons = []

        # Low confidence drop when masking important regions
        if metrics['confidence_drop_pct'] < 10:
            reasons.append("Low confidence drop (<10%) when masking important regions")

        # Significant confidence decrease when masking irrelevant regions
        if metrics['confidence_increase_pct'] < -20:
            reasons.append("Confidence decreased >20% when masking irrelevant regions")

        # Very diffuse heatmap (model uncertain)
        if metrics['concentration'] < 0.25:
            reasons.append("Very diffuse heatmap (concentration <0.25)")

        # Extremely high coverage (poor discrimination)
        if metrics['coverage'] > 0.75:
            reasons.append("Very high coverage (>75%) - poor discrimination")

        # Extremely low coverage (overfitting to small features)
        if metrics['coverage'] < 0.05:
            reasons.append("Very low coverage (<5%) - possible overfitting")

        is_suspicious = len(reasons) > 0
        reason = "; ".join(reasons) if is_suspicious else "No issues detected"

        return is_suspicious, reason
