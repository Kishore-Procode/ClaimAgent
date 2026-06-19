"""
Image utilities — loading, EXIF extraction, quality assessment,
perceptual hashing, and photo-of-photo detection.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

from PIL import Image, ExifTags
import piexif
import imagehash

logger = logging.getLogger(__name__)


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(path: str | Path) -> Optional[Image.Image]:
    """Load a PIL Image. Returns None if path does not exist or fails."""
    path = Path(path)
    if not path.exists():
        logger.warning(f"Image not found: {path}")
        return None
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}")
        return None


def get_image_id(image_path: str | Path) -> str:
    """Extract image ID from path. e.g. '.../img_2.jpg' → 'img_2'"""
    return Path(image_path).stem


# ── EXIF / Metadata ───────────────────────────────────────────────────────────

def extract_exif(path: str | Path) -> dict:
    """
    Extract EXIF metadata from a JPEG/TIFF image.

    Returns a dict with keys:
        exif_present, timestamp, gps_present, camera_make, camera_model,
        software, width, height, raw_gps
    """
    result = {
        "exif_present": False,
        "timestamp": None,
        "gps_present": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "width": None,
        "height": None,
        "raw_gps": None,
    }

    path = Path(path)
    if not path.exists():
        return result

    # Try getting dimensions via PIL
    try:
        with Image.open(path) as img:
            result["width"], result["height"] = img.size
    except Exception:
        pass

    # Try piexif for full EXIF
    try:
        exif_dict = piexif.load(str(path))
        result["exif_present"] = True

        # Camera info from 0th IFD
        zeroth = exif_dict.get("0th", {})
        make = zeroth.get(piexif.ImageIFD.Make)
        model = zeroth.get(piexif.ImageIFD.Model)
        software = zeroth.get(piexif.ImageIFD.Software)

        result["camera_make"]  = _decode_bytes(make)
        result["camera_model"] = _decode_bytes(model)
        result["software"]     = _decode_bytes(software)

        # Timestamp from Exif IFD
        exif_ifd = exif_dict.get("Exif", {})
        dt_orig = exif_ifd.get(piexif.ExifIFD.DateTimeOriginal)
        if dt_orig:
            result["timestamp"] = _decode_bytes(dt_orig)

        # GPS
        gps_ifd = exif_dict.get("GPS", {})
        if gps_ifd:
            result["gps_present"] = True
            result["raw_gps"] = {str(k): str(v) for k, v in gps_ifd.items()}

    except Exception as e:
        logger.debug(f"EXIF extraction failed for {path}: {e}")

    return result


def compute_authenticity_score(exif_data: dict) -> tuple[float, list[str]]:
    """
    Compute a basic authenticity score from EXIF data.

    Returns:
        (score 0–100, list of flag strings)
    """
    score = 100.0
    flags: list[str] = []

    if not exif_data.get("exif_present"):
        score -= 30
        flags.append("no_exif_data")

    if not exif_data.get("camera_model"):
        score -= 15

    if not exif_data.get("timestamp"):
        score -= 20

    # Software edited hint
    software = exif_data.get("software") or ""
    edited_keywords = ["photoshop", "lightroom", "gimp", "canva", "snapseed", "picsart"]
    if any(kw in software.lower() for kw in edited_keywords):
        score -= 25
        flags.append("possible_manipulation")

    score = max(0.0, min(100.0, score))
    return score, flags


# ── Perceptual Hashing ────────────────────────────────────────────────────────

def compute_phash(path: str | Path) -> Optional[imagehash.ImageHash]:
    """Compute perceptual hash for duplicate detection."""
    img = load_image(path)
    if img is None:
        return None
    try:
        return imagehash.phash(img)
    except Exception as e:
        logger.debug(f"Hash failed for {path}: {e}")
        return None


def images_are_similar(
    hash1: imagehash.ImageHash,
    hash2: imagehash.ImageHash,
    threshold: int = 8,
) -> bool:
    """Return True if two images are perceptually similar."""
    return (hash1 - hash2) <= threshold


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_bytes(value) -> Optional[str]:
    """Safely decode bytes or return string value."""
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace").strip("\x00").strip()
        except Exception:
            return str(value)
    return str(value).strip()


def resolve_image_paths(
    raw_paths_str: str, data_dir: str | Path
) -> list[Path]:
    """Split semicolon-separated paths and resolve to absolute paths."""
    data_dir = Path(data_dir)
    paths = []
    for p in raw_paths_str.split(";"):
        p = p.strip()
        if p:
            paths.append(data_dir / p)
    return paths
