import argparse
from pathlib import Path

import cv2
import numpy as np

from .utils import read_image_any_path, save_image_any_path
from . import preprocessing, denoising, features, untangle


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def cmd_preprocess(args: argparse.Namespace) -> int:
    img = read_image_any_path(args.input)
    if img is None:
        print(f"Failed to read: {args.input}")
        return 2
    gray = preprocessing.to_gray(img)
    if args.otsu:
        binary = preprocessing.to_binary(gray, use_otsu=True)
    else:
        binary = preprocessing.to_binary(gray, thresh=args.thresh, use_otsu=False)
    fourier = preprocessing.fourier_transform(gray)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_image_any_path(str(outdir / "gray.png"), gray)
    save_image_any_path(str(outdir / "binary.png"), binary)
    save_image_any_path(str(outdir / "fourier.png"), fourier)
    print(f"Saved preprocess outputs to: {outdir}")
    return 0


def cmd_denoise(args: argparse.Namespace) -> int:
    img = read_image_any_path(args.input)
    if img is None:
        print(f"Failed to read: {args.input}")
        return 2
    mask = denoising.denoise(img)
    if not save_image_any_path(args.output, mask):
        print(f"Failed to save: {args.output}")
        return 3
    print(f"Saved: {args.output}")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    img = read_image_any_path(args.input)
    if img is None:
        print(f"Failed to read: {args.input}")
        return 2
    typ = args.type.lower()
    if typ == "lbp":
        gray = _ensure_gray(img)
        out = features.extract_lbp(gray)
    elif typ == "hog":
        gray = _ensure_gray(img)
        out = features.extract_hog(gray)
    elif typ == "haar":
        out = features.extract_haar(img)
    else:
        print(f"Unknown feature type: {args.type}")
        return 2
    if not save_image_any_path(args.output, out):
        print(f"Failed to save: {args.output}")
        return 3
    print(f"Saved: {args.output}")
    return 0


def cmd_local_stats(args: argparse.Namespace) -> int:
    img = read_image_any_path(args.input)
    if img is None:
        print(f"Failed to read: {args.input}")
        return 2
    gray = _ensure_gray(img)
    win = max(3, args.win // 2 * 2 + 1)
    mean, m1, m2, ent = features.local_statistics(gray, window_size=win)
    out = np.hstack([mean, m1, m2, ent])
    if not save_image_any_path(args.output, out):
        print(f"Failed to save: {args.output}")
        return 3
    print(f"Saved: {args.output}")
    return 0


def cmd_untangle(args: argparse.Namespace) -> int:
    img = read_image_any_path(args.input)
    if img is None:
        print(f"Failed to read: {args.input}")
        return 2
    out = untangle.colorize_lines(img)
    if not save_image_any_path(args.output, out):
        print(f"Failed to save: {args.output}")
        return 3
    print(f"Saved: {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clown-vision-cli", description="Headless CLI for clown-vision")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preprocess", help="Run grayscale/binary/fourier")
    sp.add_argument("--input", required=True, help="Input image path")
    sp.add_argument("--output-dir", required=True, help="Output directory")
    sp.add_argument("--thresh", type=int, default=128, help="Binary threshold")
    sp.add_argument("--otsu", action="store_true", help="Use Otsu threshold")
    sp.set_defaults(func=cmd_preprocess)

    sp = sub.add_parser("denoise", help="Produce foreground mask (white)")
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=cmd_denoise)

    sp = sub.add_parser("features", help="Extract features: lbp/hog/haar")
    sp.add_argument("--input", required=True)
    sp.add_argument("--type", required=True, choices=["lbp", "hog", "haar"])
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=cmd_features)

    sp = sub.add_parser("local-stats", help="Compute local statistics panel")
    sp.add_argument("--input", required=True)
    sp.add_argument("--win", type=int, default=15)
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=cmd_local_stats)

    sp = sub.add_parser("untangle", help="Colorize three strings")
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=cmd_untangle)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

