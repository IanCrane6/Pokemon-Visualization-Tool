import re
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download
from PIL import Image

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR.parent / "Dataset"

def normalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def from_hugging_face():
    roster_path = BASE_DIR / "Pokemon Snap Rosters/Pokemon Snap Roster.txt"
    roster_raw = [line.strip() for line in roster_path.read_text().splitlines() if line.strip()]
    roster = {normalize(name): name for name in roster_raw}

    raw_path = Path(snapshot_download(repo_id="fcakyon/pokemon-classification", repo_type="dataset"))

    extract_root = raw_path / "extracted"
    for zip_path in (raw_path / "data").glob("*.zip"):
        extract_dest = extract_root / zip_path.stem
        if not extract_dest.exists():
            print(f"Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dest)

    dest_root = DATASET_DIR / "HuggingFace"
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

    for pokemon_folder in extract_root.rglob("*"):
        if not pokemon_folder.is_dir():
            continue

        key = normalize(pokemon_folder.name)
        if key not in roster:
            continue

        dest = dest_root / roster[key]
        dest.mkdir(parents=True, exist_ok=True)

        for img_path in pokemon_folder.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            target = dest / img_path.name
            if not target.exists():
                Image.open(img_path).save(target)