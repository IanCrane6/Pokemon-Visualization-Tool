import re
import kagglehub
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR.parent / "Dataset"

def normalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]", "", name)
    return name

def from_kaggle():
    roster_path = BASE_DIR / "Pokemon Snap Rosters/Pokemon Snap Roster.txt"
    roster_raw = [line.strip() for line in roster_path.read_text().splitlines() if line.strip()]
    roster = {normalize(name): name for name in roster_raw}

    path = Path(kagglehub.dataset_download("noodulz/pokemon-dataset-1000"))
    dest_root = DATASET_DIR / "Kaggle"

    for container in path.iterdir():

        for folders in container.iterdir():
            if folders.name != "dataset":
                continue

            for pokemon_folder in folders.iterdir():
                if not pokemon_folder.is_dir():
                    continue

                key = normalize(pokemon_folder.name)

                if key not in roster:
                    continue

                dest = dest_root / roster[key]
                dest.mkdir(parents=True, exist_ok=True)

                for img_path in pokemon_folder.rglob("*"):
                    if img_path.is_file():
                        shutil.copy2(img_path, dest / img_path.name)
