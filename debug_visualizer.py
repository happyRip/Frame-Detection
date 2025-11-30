"""Debug visualization utilities for frame detection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from models import FrameBounds, Line, Margins


class DebugVisualizer:
    """Saves debug images at each step of frame detection."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            # Backup existing debug dir before cleaning
            backup_dir = self.output_dir.with_suffix(".bak")
            if backup_dir.exists():
                import shutil

                shutil.rmtree(backup_dir)
            self.output_dir.rename(backup_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0

    def _save(self, name: str, img: np.ndarray):
        self.step += 1
        filename = f"{self.step:02d}_{name}.png"
        cv2.imwrite(str(self.output_dir / filename), img)

    def save_normalize_levels(
        self,
        img_before: np.ndarray,
        img_after: np.ndarray,
        black_point: int,
        white_point: int,
    ):
        """Save before/after normalization with histograms."""
        import matplotlib.pyplot as plt

        gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)

        hist_before = cv2.calcHist([gray_before], [0], None, [256], [0, 256]).flatten()
        hist_after = cv2.calcHist([gray_after], [0], None, [256], [0, 256]).flatten()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Before image
        axes[0, 0].imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Before normalization")
        axes[0, 0].axis("off")

        # Before histogram
        axes[0, 1].fill_between(range(256), hist_before, alpha=0.7)
        axes[0, 1].axvline(x=black_point, color="blue", linestyle="--", label=f"black={black_point}")
        axes[0, 1].axvline(x=white_point, color="red", linestyle="--", label=f"white={white_point}")
        axes[0, 1].set_xlabel("Brightness")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_xlim(0, 255)
        axes[0, 1].set_title("Histogram (before)")
        axes[0, 1].legend()

        # After image
        axes[1, 0].imshow(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("After normalization")
        axes[1, 0].axis("off")

        # After histogram
        axes[1, 1].fill_between(range(256), hist_after, alpha=0.7, color="green")
        axes[1, 1].set_xlabel("Brightness")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_xlim(0, 255)
        axes[1, 1].set_title("Histogram (after)")

        fig.tight_layout()
        self.step += 1
        fig.savefig(self.output_dir / f"{self.step:02d}_normalize_levels.png", dpi=100)
        plt.close(fig)

    def save_sprocket_holes(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        histogram: np.ndarray | None = None,
        threshold: int | None = None,
    ):
        """Save image with detected sprocket holes and histogram visualization."""
        vis = img.copy()
        # Overlay sprocket hole mask in red
        overlay = vis.copy()
        overlay[mask > 0] = (0, 0, 255)
        cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
        # Draw contours around detected holes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
        self._save("sprocket_holes", vis)

        # Save histogram visualization if provided
        if histogram is not None:
            import matplotlib.pyplot as plt
            import pandas as pd

            df = pd.DataFrame({"brightness": range(256), "count": histogram})
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.fill_between(df["brightness"], df["count"], alpha=0.7)
            ax.set_xlabel("Brightness")
            ax.set_ylabel("Count")
            ax.set_xlim(0, 255)
            if threshold is not None:
                ax.axvline(x=threshold, color="red", linestyle="--", label=f"threshold={threshold}")
                ax.legend()
            fig.tight_layout()
            fig.savefig(self.output_dir / f"{self.step + 1:02d}_sprocket_histogram.png", dpi=100)
            plt.close(fig)
            self.step += 1

    def save_sprocket_crop(
        self,
        img: np.ndarray,
        img_cropped: np.ndarray,
        y_offset_top: int,
        y_offset_bottom: int,
    ):
        """Save visualization of sprocket region cropping.

        Args:
            img: Original image
            img_cropped: Image after cropping sprocket regions
            y_offset_top: Pixels cropped from top
            y_offset_bottom: Pixels cropped from bottom
        """
        vis = img.copy()
        img_h, img_w = img.shape[:2]

        if y_offset_top > 0 or y_offset_bottom > 0:
            overlay = vis.copy()
            # Shade cropped top region
            if y_offset_top > 0:
                overlay[:y_offset_top, :] = (0, 0, 128)  # Dark red overlay
                cv2.line(vis, (0, y_offset_top), (img_w, y_offset_top), (0, 255, 255), 3)
            # Shade cropped bottom region
            if y_offset_bottom > 0:
                crop_bottom = img_h - y_offset_bottom
                overlay[crop_bottom:, :] = (0, 0, 128)  # Dark red overlay
                cv2.line(vis, (0, crop_bottom), (img_w, crop_bottom), (0, 255, 255), 3)
            cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
            cv2.putText(vis, "SPROCKET REGIONS", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        else:
            cv2.putText(vis, "No sprocket crop", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        self._save("sprocket_crop", vis)

    def save_film_base(
        self,
        img: np.ndarray,
        sample_mask: np.ndarray,
        film_base: np.ndarray,
        from_sprocket_regions: bool,
    ):
        """Save visualization of film base color detection.

        Args:
            img: Original image
            sample_mask: Mask showing sampled regions (255=sampled)
            film_base: Detected film base color (BGR)
            from_sprocket_regions: Whether samples came from sprocket regions
        """
        vis = img.copy()
        img_h, img_w = img.shape[:2]

        # Overlay sampled regions in cyan
        overlay = vis.copy()
        overlay[sample_mask > 0] = (255, 255, 0)  # Cyan overlay for sampled areas
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

        # Draw a color swatch showing the detected film base color
        swatch_size = max(80, min(img_h, img_w) // 8)
        swatch_x = img_w - swatch_size - 20
        swatch_y = 20
        cv2.rectangle(
            vis,
            (swatch_x, swatch_y),
            (swatch_x + swatch_size, swatch_y + swatch_size),
            film_base.tolist(),
            -1,
        )
        cv2.rectangle(
            vis,
            (swatch_x, swatch_y),
            (swatch_x + swatch_size, swatch_y + swatch_size),
            (255, 255, 255),
            3,
        )

        # Add labels
        source = "sprocket regions" if from_sprocket_regions else "image edges"
        cv2.putText(
            vis,
            f"Film base (from {source})",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
        )

        b, g, r = film_base
        cv2.putText(
            vis,
            f"BGR: ({b}, {g}, {r})",
            (swatch_x - 100, swatch_y + swatch_size + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        self._save("film_base", vis)

    def save_film_base_mask(
        self,
        img: np.ndarray,
        film_base_mask: np.ndarray,
        film_base_color: np.ndarray,
        tolerance: int,
    ):
        """Save visualization of film base mask.

        Args:
            img: Original image
            film_base_mask: Binary mask where film base regions are 255
            film_base_color: The detected film base color (BGR)
            tolerance: Color tolerance used for matching
        """
        vis = img.copy()
        img_h, img_w = img.shape[:2]

        # Overlay film base mask in magenta
        overlay = vis.copy()
        overlay[film_base_mask > 0] = (255, 0, 255)  # Magenta for film base regions
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

        # Draw contours around film base regions
        contours, _ = cv2.findContours(
            film_base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

        # Draw color swatch
        swatch_size = max(80, min(img_h, img_w) // 8)
        swatch_x = img_w - swatch_size - 20
        swatch_y = 20
        cv2.rectangle(
            vis,
            (swatch_x, swatch_y),
            (swatch_x + swatch_size, swatch_y + swatch_size),
            film_base_color.tolist(),
            -1,
        )
        cv2.rectangle(
            vis,
            (swatch_x, swatch_y),
            (swatch_x + swatch_size, swatch_y + swatch_size),
            (255, 255, 255),
            3,
        )

        # Add labels
        cv2.putText(
            vis,
            f"Film base mask (tolerance={tolerance})",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
        )

        self._save("film_base_mask", vis)

    def save_film_base_mask_cropped(
        self,
        img: np.ndarray,
        film_base_mask_cropped: np.ndarray,
        y_min: int,
        y_max: int,
    ):
        """Save visualization of film base mask with sprocket areas excluded.

        Args:
            img: Original full image
            film_base_mask_cropped: Binary mask cropped to valid region only
            y_min: Top boundary of valid frame region
            y_max: Bottom boundary of valid frame region
        """
        vis = img.copy()
        img_h, img_w = img.shape[:2]

        # Shade excluded sprocket regions in dark red
        if y_min > 0:
            vis[:y_min, :] = (vis[:y_min, :] * 0.3 + np.array([0, 0, 100])).astype(
                np.uint8
            )
        if y_max < img_h:
            vis[y_max:, :] = (vis[y_max:, :] * 0.3 + np.array([0, 0, 100])).astype(
                np.uint8
            )

        # Overlay film base mask in magenta (in valid region)
        overlay = vis.copy()
        overlay[y_min:y_max, :][film_base_mask_cropped > 0] = (255, 0, 255)
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

        # Draw boundary lines for valid region
        cv2.line(vis, (0, y_min), (img_w, y_min), (0, 255, 255), 2)
        cv2.line(vis, (0, y_max), (img_w, y_max), (0, 255, 255), 2)

        cv2.putText(
            vis,
            "Film base mask (sprocket areas excluded)",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
        )

        self._save("film_base_mask_cropped", vis)

    def save_edges(self, edges: np.ndarray):
        """Save edge detection result."""
        self._save("edges", edges)

    def save_edges_variations(self, blurred: np.ndarray, median: float, current_low_factor: float, current_high_factor: float):
        """Save multiple Canny edge detection results with varying parameters.

        Args:
            blurred: Blurred grayscale image for edge detection
            median: Median value of the blurred image
            current_low_factor: Currently used low threshold factor
            current_high_factor: Currently used high threshold factor
        """
        import matplotlib.pyplot as plt

        # Factor variations to try (low_factor, high_factor, label)
        variations = [
            (0.33, 0.66, "0.33/0.66"),
            (0.5, 1.0, "0.5/1.0"),
            (current_low_factor, current_high_factor, f"current ({current_low_factor}/{current_high_factor})"),
            (0.8, 1.5, "0.8/1.5"),
            (1.0, 2.0, "1.0/2.0"),
            (1.33, 2.66, "1.33/2.66"),
        ]

        # Create a grid of edge images
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (low_f, high_f, label) in enumerate(variations):
            low = int(max(0, low_f * median))
            high = int(min(255, high_f * median))
            edges = cv2.Canny(blurred, low, high)
            axes[idx].imshow(edges, cmap="gray")
            axes[idx].set_title(f"{label}\nlow={low}, high={high}")
            axes[idx].axis("off")

        fig.suptitle(f"Canny Edge Detection (median={median:.0f})", fontsize=14)
        fig.tight_layout()
        self.step += 1
        fig.savefig(self.output_dir / f"{self.step:02d}_edges_variations.png", dpi=150)
        plt.close(fig)

    def save_lines(self, img: np.ndarray, lines: list[Line]):
        """Save image with all detected lines drawn."""
        vis = img.copy()
        for line in lines:
            cv2.line(vis, (line.x1, line.y1), (line.x2, line.y2), (0, 255, 0), 2)
        self._save("lines_all", vis)

    def save_ignore_margin(
        self,
        img: np.ndarray,
        ignore_margins: Margins,
    ):
        """Save visualization of ignored margins.

        Args:
            img: Original image before cropping
            ignore_margins: Margins object with (top, right, bottom, left) fractions
        """
        vis = img.copy()
        img_h, img_w = img.shape[:2]

        ignore_top = int(img_h * ignore_margins.top)
        ignore_bottom = int(img_h * ignore_margins.bottom)
        ignore_left = int(img_w * ignore_margins.left)
        ignore_right = int(img_w * ignore_margins.right)

        overlay = vis.copy()
        # Shade ignored regions in red
        if ignore_top > 0:
            overlay[:ignore_top, :] = (0, 0, 128)
        if ignore_bottom > 0:
            overlay[img_h - ignore_bottom :, :] = (0, 0, 128)
        if ignore_left > 0:
            overlay[:, :ignore_left] = (0, 0, 128)
        if ignore_right > 0:
            overlay[:, img_w - ignore_right :] = (0, 0, 128)

        cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

        # Draw analysis region rectangle in green
        cv2.rectangle(
            vis,
            (ignore_left, ignore_top),
            (img_w - ignore_right, img_h - ignore_bottom),
            (0, 255, 0),
            3,
        )

        cv2.putText(vis, "Ignore margins (red=ignored)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        self._save("ignore_margin", vis)

    def save_edge_margins(
        self,
        img: np.ndarray,
        edge_margins: Margins,
        y_min: int = 0,
        y_max: int | None = None,
    ):
        """Save visualization of edge margin zones for line detection.

        Args:
            img: Image being analyzed
            edge_margins: Margins object with (top, right, bottom, left) fractions
            y_min: Minimum valid y coordinate (after sprocket cropping)
            y_max: Maximum valid y coordinate (after sprocket cropping)
        """
        vis = img.copy()
        img_h, img_w = img.shape[:2]

        if y_max is None:
            y_max = img_h

        # Apply edge margins relative to the valid region (after sprocket cropping)
        valid_height = y_max - y_min
        y_top = y_min + int(valid_height * edge_margins.top)
        y_bottom = y_min + int(valid_height * (1 - edge_margins.bottom))
        x_left = int(img_w * edge_margins.left)
        x_right = int(img_w * (1 - edge_margins.right))

        overlay = vis.copy()
        # Shade edge zones in green (where lines are considered)
        overlay[:y_top, :] = (0, 128, 0)  # Top zone
        overlay[y_bottom:, :] = (0, 128, 0)  # Bottom zone
        overlay[:, :x_left] = (0, 128, 0)  # Left zone
        overlay[:, x_right:] = (0, 128, 0)  # Right zone

        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

        # Draw zone boundaries
        cv2.line(vis, (0, y_top), (img_w, y_top), (0, 255, 0), 2)
        cv2.line(vis, (0, y_bottom), (img_w, y_bottom), (0, 255, 0), 2)
        cv2.line(vis, (x_left, 0), (x_left, img_h), (0, 255, 0), 2)
        cv2.line(vis, (x_right, 0), (x_right, img_h), (0, 255, 0), 2)

        cv2.putText(vis, "Edge margins (green=detection zones)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        self._save("edge_margins", vis)

    def save_classified_lines(
        self,
        img: np.ndarray,
        frame_bounds: FrameBounds,
    ):
        """Save image with lines colored by edge classification."""
        vis = img.copy()

        # Define colors for each edge
        colors = {
            "top": (255, 0, 0),      # Blue
            "bottom": (255, 128, 0),  # Light blue
            "left": (0, 0, 255),      # Red
            "right": (0, 128, 255),   # Orange
        }

        # Draw lines for each edge group
        for edge_name, edge_group in [
            ("top", frame_bounds.top),
            ("bottom", frame_bounds.bottom),
            ("left", frame_bounds.left),
            ("right", frame_bounds.right),
        ]:
            color = colors[edge_name]
            for line in edge_group.lines:
                cv2.line(vis, (line.x1, line.y1), (line.x2, line.y2), color, 2)

        # Add legend
        y_offset = 30
        for edge_name, color in colors.items():
            count = len(getattr(frame_bounds, edge_name).lines)
            cv2.putText(vis, f"{edge_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += 30

        self._save("lines_classified", vis)

    def save_crop_in(
        self,
        img: np.ndarray,
        bounds_before: list[int],
        bounds_after: list[int],
        crop_in_percent: float,
    ):
        """Save visualization of crop-in adjustment.

        Args:
            img: Input image
            bounds_before: Bounds before crop-in [left, right, top, bottom]
            bounds_after: Bounds after crop-in [left, right, top, bottom]
            crop_in_percent: Percentage cropped from each edge
        """
        LEFT, RIGHT, TOP, BOTTOM = 0, 1, 2, 3
        vis = img.copy()

        # Draw original bounds in red (solid)
        cv2.rectangle(
            vis,
            (bounds_before[LEFT], bounds_before[TOP]),
            (bounds_before[RIGHT], bounds_before[BOTTOM]),
            (0, 0, 255),
            3,
        )

        # Draw cropped bounds in green (solid)
        cv2.rectangle(
            vis,
            (bounds_after[LEFT], bounds_after[TOP]),
            (bounds_after[RIGHT], bounds_after[BOTTOM]),
            (0, 255, 0),
            3,
        )

        # Add legend
        cv2.putText(vis, f"Crop-in: {crop_in_percent:.1f}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(vis, "Before", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(vis, "After", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        self._save("crop_in", vis)

    def save_bounds(self, img: np.ndarray, bounds: list[int]):
        """Save image with detected frame bounds rectangle.

        Args:
            img: Input image
            bounds: Array of [left, right, top, bottom] positions
        """
        LEFT, RIGHT, TOP, BOTTOM = 0, 1, 2, 3
        vis = img.copy()
        cv2.rectangle(vis, (bounds[LEFT], bounds[TOP]),
                      (bounds[RIGHT], bounds[BOTTOM]), (0, 255, 0), 3)
        self._save("bounds_final", vis)
