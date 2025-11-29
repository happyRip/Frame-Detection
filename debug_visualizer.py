"""Debug visualization utilities for frame detection."""

from pathlib import Path

import cv2
import numpy as np


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

    def save_edges(self, edges: np.ndarray):
        """Save Canny edge detection result."""
        self._save("edges_canny", edges)

    def save_lines(self, img: np.ndarray, lines: np.ndarray):
        """Save image with all detected lines drawn."""
        vis = img.copy()
        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self._save("lines_all", vis)

    def save_classified_lines(
        self,
        img: np.ndarray,
        lines: np.ndarray,
        horiz_positions: list[int],
        vert_positions: list[int],
    ):
        """Save image with lines colored by classification."""
        vis = img.copy()
        for x1, y1, x2, y2 in lines[:, 0, :]:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 20:  # horizontal - blue
                cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif abs(angle - 90) < 20 or abs(angle + 90) < 20:  # vertical - red
                cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:  # other - gray
                cv2.line(vis, (x1, y1), (x2, y2), (128, 128, 128), 1)

        # Draw detected edge positions as dotted lines
        for y in horiz_positions:
            for x in range(0, vis.shape[1], 10):
                cv2.circle(vis, (x, y), 1, (255, 100, 0), -1)
        for x in vert_positions:
            for y in range(0, vis.shape[0], 10):
                cv2.circle(vis, (x, y), 1, (0, 100, 255), -1)

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
