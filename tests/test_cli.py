import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from clown_vision.cli import main as cli_main


ASSETS = Path("assets")
OUT = Path("assets/cli_out")
OUT.mkdir(parents=True, exist_ok=True)


def test_preprocess_cli():
    inp = str(ASSETS / "test.png")
    code = cli_main(["preprocess", "--input", inp, "--output-dir", str(OUT), "--otsu"])
    assert code == 0
    for name in ["gray.png", "binary.png", "fourier.png"]:
        p = OUT / name
        assert p.exists()
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        assert img is not None


def test_denoise_cli():
    inp = str(ASSETS / "test.png")
    outp = str(OUT / "denoise.png")
    code = cli_main(["denoise", "--input", inp, "--output", outp])
    assert code == 0
    img = cv2.imread(outp, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    # Foreground should have some white pixels
    assert int((img > 0).sum()) > 100


def test_features_cli():
    inp = str(ASSETS / "test.png")
    for typ in ("lbp", "hog", "haar"):
        outp = str(OUT / f"feat_{typ}.png")
        code = cli_main(["features", "--input", inp, "--type", typ, "--output", outp])
        assert code == 0
        img = cv2.imread(outp, cv2.IMREAD_UNCHANGED)
        assert img is not None


def test_local_stats_cli():
    inp = str(ASSETS / "test.png")
    outp = str(OUT / "local_stats.png")
    code = cli_main(["local-stats", "--input", inp, "--win", "15", "--output", outp])
    assert code == 0
    img = cv2.imread(outp, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    h, w = img.shape
    # The hstack of 4 panels should have width roughly 4x gray width; we only check existence
    assert h > 0 and w > 0


def test_untangle_cli():
    inp = str(ASSETS / "test.png")
    outp = str(OUT / "untangle.png")
    code = cli_main(["untangle", "--input", inp, "--output", outp])
    assert code == 0
    img = cv2.imread(outp, cv2.IMREAD_COLOR)
    assert img is not None

