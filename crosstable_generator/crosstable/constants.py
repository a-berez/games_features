from pathlib import Path

BASE_URL = "https://api.rating.chgk.net"
MIRROR_URL = "https://rating-api.pecheny.me"

CELL_SEP = " — "
DIAGONAL = "×"
CELL_PARTS = ("x", "y", "z", "w")

STATIC_FILES = ("crosstable.css", "crosstable.js")

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PACKAGE_ROOT / "templates"
STATIC_DIR = PACKAGE_ROOT / "static"
