"""Command-line interface for frame detection."""

import argparse
import sys
from pathlib import Path

DEFAULT_DELIM = "_"


def write_error(output_path: str | None, message: str) -> None:
    """Write error message to error file for plugin to read."""
    if output_path:
        error_path = output_path + ".err"
        with open(error_path, "w") as f:
            f.write(message)


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


def add_detect_arguments(parser: argparse.ArgumentParser) -> None:
    """Add frame detection arguments to a parser."""
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
        "--sprocket-margin",
        type=float,
        default=0.1,
        help="Percentage to crop beyond detected sprocket holes (0-100)",
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
        "--film-base-inset",
        type=float,
        default=1.0,
        help="Diagonal inset percentage for film base sampling region (0-50, used when no sprockets)",
    )
    parser.add_argument(
        "--coords",
        action="store_true",
        help="Output crop coordinates (0.0-1.0) and angle to text file instead of cropped image",
    )
    parser.add_argument(
        "--film-type",
        choices=["auto", "negative", "positive"],
        default="auto",
        help="Film type: 'negative' (bright sprocket holes), 'positive' (dark sprocket holes), "
        "or 'auto' to detect automatically (default: auto)",
    )
    parser.add_argument(
        "--edge-filter",
        choices=["canny", "sobel", "scharr", "dog", "laplacian", "log"],
        default="scharr",
        help="Edge detection filter: canny, sobel, scharr (default), dog, laplacian, or log",
    )
    parser.add_argument(
        "--separation-method",
        choices=["color_distance", "clahe", "lab_distance", "hsv_distance", "adaptive", "gradient"],
        default="color_distance",
        help="Film base separation method: color_distance (default), clahe, lab_distance, "
        "hsv_distance, adaptive, or gradient",
    )
    parser.add_argument(
        "--filter-config",
        help="JSON filter configuration (inline JSON string or path to .json file). "
        "Overrides --edge-filter and --separation-method if provided.",
    )
    parser.add_argument(
        "--preview-mode",
        choices=["separation", "edges"],
        help="Generate minimal preview: 'separation' for film base mask, 'edges' for edge detection. "
        "Exits early after generating the requested visualization.",
    )


def parse_filter_config(config_arg: str | None):
    """Parse filter config from CLI argument.

    Args:
        config_arg: Either inline JSON string or path to .json file

    Returns:
        FilterConfig object or None if not provided
    """
    if not config_arg:
        return None

    from .models import FilterConfig

    # Check if it looks like a file path
    config_path = Path(config_arg)
    if config_path.exists() and config_path.suffix == ".json":
        return FilterConfig.from_file(config_path)

    # Try parsing as inline JSON
    try:
        return FilterConfig.from_json(config_arg)
    except Exception as e:
        raise ValueError(f"Invalid --filter-config: {e}")


def run_detect(args: argparse.Namespace) -> None:
    """Run frame detection on an image."""
    import cv2

    from .detection import crop_frame, detect_frame_bounds
    from .exceptions import FrameDetectionError, ImageReadError
    from .filters import EdgeFilter
    from .models import FilmType, Margins
    from .separation import SeparationMethod

    # Determine error output path (used if --output is set)
    error_output = args.output if args.output else None

    img = cv2.imread(args.input)
    if img is None:
        err = ImageReadError(args.input)
        write_error(error_output, err.user_message)
        sys.exit(err.user_message)

    img_h, img_w = img.shape[:2]
    landscape = img_w >= img_h
    aspect_ratio = parse_aspect_ratio(args.ratio, landscape)

    visualizer = None
    if args.debug_dir:
        from .visualizer import DebugVisualizer

        visualizer = DebugVisualizer(args.debug_dir)

    edge_margins = Margins.parse(args.edge_margin)
    ignore_margins = Margins.parse(args.ignore_margin)

    # Parse film type
    film_type_map = {
        "auto": FilmType.AUTO,
        "negative": FilmType.NEGATIVE,
        "positive": FilmType.POSITIVE,
    }
    film_type = film_type_map[args.film_type]

    # Parse edge filter
    edge_filter_map = {
        "canny": EdgeFilter.CANNY,
        "sobel": EdgeFilter.SOBEL,
        "scharr": EdgeFilter.SCHARR,
        "dog": EdgeFilter.DOG,
        "laplacian": EdgeFilter.LAPLACIAN,
        "log": EdgeFilter.LOG,
    }
    edge_filter = edge_filter_map[args.edge_filter]

    # Parse separation method
    separation_method_map = {
        "color_distance": SeparationMethod.COLOR_DISTANCE,
        "clahe": SeparationMethod.CLAHE,
        "lab_distance": SeparationMethod.LAB_DISTANCE,
        "hsv_distance": SeparationMethod.HSV_DISTANCE,
        "adaptive": SeparationMethod.ADAPTIVE,
        "gradient": SeparationMethod.GRADIENT,
    }
    separation_method = separation_method_map[args.separation_method]

    # Parse filter config (overrides edge_filter and separation_method if provided)
    filter_config = parse_filter_config(getattr(args, "filter_config", None))

    # Keep a copy of original image for debug visualization
    img_original = img.copy() if visualizer else None

    # Parse preview mode
    preview_mode = getattr(args, "preview_mode", None)

    try:
        frame_bounds, bounds = detect_frame_bounds(
            img,
            aspect_ratio,
            visualizer=visualizer,
            crop_in_percent=args.crop_in,
            sprocket_margin_percent=args.sprocket_margin,
            edge_margins=edge_margins,
            ignore_margins=ignore_margins,
            film_base_inset_percent=args.film_base_inset,
            film_type=film_type,
            edge_filter=edge_filter,
            separation_method=separation_method,
            filter_config=filter_config,
            preview_mode=preview_mode,
        )
    except FrameDetectionError as e:
        write_error(error_output, e.user_message)
        sys.exit(e.user_message)
    except Exception as e:
        msg = f"Unexpected error: {e}"
        write_error(error_output, msg)
        sys.exit(msg)

    # Early exit for preview modes - visualization already saved, no further processing needed
    if preview_mode:
        return

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


def run_install_shortcuts(args: argparse.Namespace) -> None:
    """Install keyboard shortcuts."""
    from .shortcuts import install_shortcuts

    success = install_shortcuts()
    sys.exit(0 if success else 1)


def run_uninstall_shortcuts(args: argparse.Namespace) -> None:
    """Uninstall keyboard shortcuts."""
    from .shortcuts import uninstall_shortcuts

    success = uninstall_shortcuts()
    sys.exit(0 if success else 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Film negative frame detection and cropping tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  negative-auto-crop detect image.jpg                  Detect and crop frame
  negative-auto-crop detect image.jpg --coords -o out.txt  Output coordinates
  negative-auto-crop install shortcuts          Install Lightroom shortcuts
  negative-auto-crop uninstall shortcuts        Remove Lightroom shortcuts
""",
    )
    subparsers = parser.add_subparsers(dest="command")

    # detect subcommand (explicit)
    detect_parser = subparsers.add_parser(
        "detect",
        help="Detect and crop frame from image",
    )
    add_detect_arguments(detect_parser)
    detect_parser.set_defaults(func=run_detect)

    # install subcommand
    install_parser = subparsers.add_parser("install", help="Install components")
    install_subparsers = install_parser.add_subparsers(dest="target")
    shortcuts_install = install_subparsers.add_parser(
        "shortcuts",
        help="Install Lightroom keyboard shortcuts (macOS only)",
    )
    shortcuts_install.set_defaults(func=run_install_shortcuts)

    # uninstall subcommand
    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall components")
    uninstall_subparsers = uninstall_parser.add_subparsers(dest="target")
    shortcuts_uninstall = uninstall_subparsers.add_parser(
        "shortcuts",
        help="Remove Lightroom keyboard shortcuts (macOS only)",
    )
    shortcuts_uninstall.set_defaults(func=run_uninstall_shortcuts)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)
    elif hasattr(args, "func"):
        args.func(args)
    else:
        # Subcommand without target (e.g., "install" without "shortcuts")
        if args.command == "install":
            install_parser.print_help()
        elif args.command == "uninstall":
            uninstall_parser.print_help()
        else:
            parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
