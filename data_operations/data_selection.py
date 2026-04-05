import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATASET_DIR = Path(__file__).parent.parent / "Dataset"
ROSTERS_DIR = Path(__file__).parent / "Pokemon Snap Rosters"

ALL_SOURCES = ["PokemonAPI", "Kaggle", "HuggingFace"]

level_dict = {
    1: ROSTERS_DIR / "Beach.txt",
    2: ROSTERS_DIR / "Tunnel.txt",
    3: ROSTERS_DIR / "Volcano.txt",
    4: ROSTERS_DIR / "River.txt",
    5: ROSTERS_DIR / "Cave.txt",
    6: ROSTERS_DIR / "Valley.txt",
    7: ROSTERS_DIR / "Rainbow.txt",
}


def load_pokemon_for_levels(levels) -> set[str]:
    if levels == "all":
        level_paths = level_dict.values()
    else:
        level_paths = [level_dict[l] for l in levels]

    pokemon = set()
    for path in level_paths:

        with open(path, "r") as f:
            for line in f:
                pokemon.add(line.strip().lower())
    return pokemon


def select_data(sources, levels) -> pd.DataFrame:
    """
    Generates the pokemon dataframe for future splitting
    :param sources: The sources to use for the data. Either "all" or some combination of "PokemonAPI", "Kaggle", "HuggingFace" in list format
    :param levels: The levels to use for the data. Either "all" or a list of levels in the range 1-7 (also in list format)
    :return: The dataframe containing the image paths, labels, and sources
    """
    if sources == "all":
        sources = ALL_SOURCES

    pokemon_set = load_pokemon_for_levels(levels)

    rows = []
    for source in sources:
        source_path = DATASET_DIR / source
        if not source_path.exists():
            print(f"Warning: {source} folder not found, skipping.")
            continue

        for img_path in source_path.rglob("*"):
            if not img_path.is_file():
                continue
            if img_path.name.lower() in pokemon_set or img_path.parent.name.lower() in pokemon_set:
                rows.append({
                    "image_path": str(img_path),
                    "label": img_path.parent.name,
                    "source": source,
                })

    df = pd.DataFrame(rows)
    print(f"Selected {len(df)} images across {df['label'].nunique()} Pokemon.")
    return df


def split_data(df: pd.DataFrame, test_val_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    A fucntion to split the data into consistent train test, and validation sets. All use the same seed for reproducibility.
    :param df: The dataframe to split (from select_data)
    :param test_val_size: The percent of the dataset that will be used to create the test and validation sets. The test and validation sets will be 50% each.
    :return: The three dataframes in train, test, validation order
    """
    train, temp = train_test_split(df, test_size=test_val_size, random_state=42, stratify=df["label"])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label"])
    return train, test, val