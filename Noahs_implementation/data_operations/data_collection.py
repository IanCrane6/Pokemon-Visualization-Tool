from pokeapi_collection import from_pokeapi
from kaggle_collection import from_kaggle
from hugging_face_collection import from_hugging_face


def build_dataset():
    print("Collecting data from PokeAPI")
    from_pokeapi()

    print("Collecting data from Kaggle")
    from_kaggle()

    print("Collecting data from HuggingFace")
    from_hugging_face()

    print("Data collection completed")


