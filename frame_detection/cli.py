"""Command-line interface for frame detection."""

import argparse
import sys
from pathlib import Path

import cv2

from .detection import crop_frame, detect_frame_bounds
from .models import Margins

DEFAULT_DELIM = "_"


def detect_delim(filename: str) -> str | None:
    """Detect delimiter from filename by finding most common separator."""
    stem = Path(filename).stem
    for delim in ["_", "-", "."]:
        if delim in stem:
            return delim
    return None


def parse_aspect_ratio(value: str, landscape: bool) -> float:
    """Parse aspect ratio string and orient based on image orientation.

    Ratios like 3/2 and 2/3 are treated as equivalent - the orientation
    is determined by the image dimensions, not the input order.
    """
    for sep in ["/", ":"]:
        if sep in value:
            a, b = map(float, value.split(sep, 1))
            break
    else:
        a, b = float(value), 1.0

    # Normalize: return width/height based on image orientation
    long, short = max(a, b), min(a, b)
    return long / short if landscape else short / long


def build_output_filename(input_path: str, prefix: str, suffix: str, delim: str) -> str:
    p = Path(input_path)
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(p.stem)
    if suffix:
        parts.append(suffix)
    return str(p.with_stem(delim.join(parts)))


def parse_args():
    parser = argparse.ArgumentParser(description="Detect and crop frame from image")
    parser.add_argument("input", help="Input image file")
    parser.add_argument("-o", "--output", help="Output filename")
    parser.add_argument("--prefix", default="", help="Prefix for output filename")
    parser.add_argument("--suffix", default="crop", help="Suffix for output filename")
    parser.add_argument(
        "--delim",
        help="Delimiter between prefix/name/suffix (auto-detected from filename if not set)",
    )
    parser.add_argument(
        "--default-delim",
        default=DEFAULT_DELIM,
        help=f"Default delimiter if not detected (default: '{DEFAULT_DELIM}')",
    )
    parser.add_argument(
        "-r",
        "--ratio",
        default="3/2",
        help="Frame aspect ratio, e.g. 3/2, 6/6, 6/4.5, 4/5 (orientation auto-detected from image)",
    )
    parser.add_argument(
        "--debug-dir",
        help="Directory to save debug visualization images",
    )
    parser.add_argument(
        "--crop-in",
        type=float,
        default=1.5,
        help="Percentage to crop inward from detected edges (0-100)",
    )
    parser.add_argument(
        "--edge-margin",
        default="5",
        help="Percentage of edge margin for line detection: single value (5), "
        "vertical,horizontal (5,10), or top,right,bottom,left (5,10,7.5,13.33)",
    )
    parser.add_argument(
        "--ignore-margin",
        default="0,1",
        help="Percentage of image margin to ignore during analysis: single value (5), "
        "vertical,horizontal (0,5), or top,right,bottom,left (0,5,0,5)",
    )
    parser.add_argument(
        "--coords",
        action="store_true",
        help="Output crop coordinates (0.0-1.0) and angle to text file instead of cropped image",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    img = cv2.imread(args.input)
    if img is None:
        sys.exit(f"Could not read image: {args.input}")

    img_h, img_w = img.shape[:2]
    landscape = img_w >= img_h
    aspect_ratio = parse_aspect_ratio(args.ratio, landscape)

    visualizer = None
    if args.debug_dir:
        from .visualizer import DebugVisualizer

        visualizer = DebugVisualizer(args.debug_dir)

    edge_margins = Margins.parse(args.edge_margin)
    ignore_margins = Margins.parse(args.ignore_margin)

    # Keep a copy of original image for debug visualization
    img_original = img.copy() if visualizer else None

    try:
        frame_bounds, bounds = detect_frame_bounds(
            img,
            aspect_ratio,
            visualizer=visualizer,
            crop_in_percent=args.crop_in,
            edge_margins=edge_margins,
            ignore_margins=ignore_margins,
        )
    except ValueError as e:
        sys.exit(str(e))

    left, right, top, bottom = bounds

    if args.coords:
        # Output coordinates as fractions (0.0-1.0) and angle
        left_frac = left / img_w
        right_frac = right / img_w
        top_frac = top / img_h
        bottom_frac = bottom / img_h
        angle = 0.0  # TODO: calculate rotation angle

        if visualizer:
            visualizer.save_coords_output(
                img_original, bounds, left_frac, right_frac, top_frac, bottom_frac
            )

        output = f"{left_frac}\n{right_frac}\n{top_frac}\n{bottom_frac}\n{angle}"
        if args.output:
            with open(args.output, "w") as f:
                f.write(output + "\n")
        else:
            print(output)
    else:
        crop = crop_frame(img, bounds)

        if args.output:
            output_path = args.output
        else:
            delim = args.delim or detect_delim(args.input) or args.default_delim
            output_path = build_output_filename(
                args.input, args.prefix, args.suffix, delim
            )

        cv2.imwrite(output_path, crop)


if __name__ == "__main__":
    main()
