import requests
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

pokemon_api = 'https://pokeapi.co/api/v2/pokemon/'
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR.parent / "Dataset"

def fetch_pokemon(pokemon_name):
    pokemon_url = pokemon_api + pokemon_name
    response = requests.get(pokemon_url)
    data = response.json()
    sprites = data['sprites']
    get_sprites(sprites, pokemon_name=pokemon_name)


def from_pokeapi():
    roster_path = BASE_DIR / 'Pokemon Snap Rosters/Pokemon Snap Roster.txt'
    with open(roster_path, 'r') as f:
        pokemon_list = [line.strip() for line in f if line.strip()]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_pokemon, name): name for name in pokemon_list}
        for future in as_completed(futures):
            if future.exception():
                print(f'{futures[future]} failed: {future.exception()}')


def get_sprites(sprites, path="", pokemon_name=""):
    for key, value in sprites.items():
        if isinstance(value, dict):
            get_sprites(value, path=f"{path}{key}/", pokemon_name=pokemon_name)
        elif isinstance(value, str):
            path = path.replace("/", "_")
            save_path = DATASET_DIR / 'PokemonAPI' / pokemon_name / f'{path}{key}.png'
            save_image(value, save_path)

def save_image(image_url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    image = requests.get(image_url, stream=True)

    with open(save_path, 'wb') as f:
        f.write(image.content)