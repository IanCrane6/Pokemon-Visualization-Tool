import shutil

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

def convert_labels_to_numeric_representation(df: pd.DataFrame, pokemon_labels: list = None) -> dict:
    """
    A function to convert the string labels to numeric labels for the yolo format. The numeric labels will be based on the order of the pokemon_labels list. If pokemon_labels is None, then the numeric labels will be based on the unique labels in the dataframe, sorted alphabetically.
    :param df: The dataframe containing the image paths, labels, and sources
    :param pokemon_labels: The list of pokemon labels to use for the numeric labels. If None, then the unique labels in the dataframe will be used.
    :return: A dictionary mapping the string labels to numeric labels
    """
    if pokemon_labels is not None:
        lower_case_labels = [label.lower() for label in pokemon_labels]
        label_to_numeric = {label: i for i, label in enumerate(lower_case_labels)}
    else:
        unique_labels = sorted(df["label"].str.lower().unique())
        label_to_numeric = {label: i for i, label in enumerate(unique_labels)}
    

    return label_to_numeric

def create_yolo_format(data: pd.DataFrame,data_type:str, pokemon_labels: dict) -> None:
    """
    A function to create the yolo format text files for the data. The text files will be created in the same directory as the images, with the same name but with a .txt extension.
    THIS WILL COPY THE IMAGES
    The txt files will contain the label, and the bounding box coordinates (which will be 0.5 0.5 1.0 1.0 since we are treating the whole image as the bounding box). The images will be copied to a new directory structure that is compatible with Ultralytics YOLO format.
    :param data: The dataframe containing the image paths, labels, and sources
    :param data_type: The type of data (train, test, val)
    :param pokemon_labels: The dictionary mapping the string labels to numeric labels.
    """
    if data_type == "train":
        print("Creating YOLO format text files for training data...")
        type_of_data = "train"
    elif data_type == "test":
        print("Creating YOLO format text files for testing data...")
        type_of_data = "test"
    elif data_type == "val":
        print("Creating YOLO format text files for validation data...")
        type_of_data = "val"
    else:
        raise ValueError("data_type must be one of 'train', 'test', or 'val'")
    
    images_path_path = ""
    labels_path_path = ""

    for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Creating yolo data for {data_type} "):
        # if pokemon_labels is not None and row["label"].lower() not in lower_case_labels:
        #     continue
        if pokemon_labels is not None and row["label"].lower() not in pokemon_labels:
            continue
        img_path = Path(row["image_path"])
        label = row["label"]
        txt_path = img_path.with_suffix(".txt")



        labels_path = f"../Dataset/My_yolo_dataset/{type_of_data}/labels/"


        # check if the labels path exists, if not create it
        labels_path_path = Path(labels_path)
        if not Path(labels_path).exists():
            labels_path_path.mkdir(parents=True, exist_ok=True)

        txt_path = f"{labels_path_path}/{txt_path.name}"
        if not Path(txt_path).exists():
            with open(txt_path, "w") as f:
                f.write(f"{pokemon_labels[label.lower()]} 0.5 0.5 1.0 1.0\n")
 
        images_path = f"../Dataset/My_yolo_dataset/{type_of_data}/images/"
        if not Path(images_path).exists():
            images_path_path = Path(images_path)
            images_path_path.mkdir(parents=True, exist_ok=True)
        else:
            images_path_path = Path(images_path)

        test_existing_path = f"{images_path_path}/{img_path.name}"
        if not Path(test_existing_path).exists():
            shutil.copy(img_path, test_existing_path)
        
