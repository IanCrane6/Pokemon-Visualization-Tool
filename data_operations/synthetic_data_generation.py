import random
import math
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from data_selection import _is_3d, load_pokemon_for_levels

np.seterr(divide='ignore', invalid='ignore')

BASE_DIR        = Path(__file__).parent.parent
DATASET_DIR     = BASE_DIR / "Dataset"
BACKGROUNDS_DIR = DATASET_DIR / "Backgrounds"
SPRITES_DIR     = DATASET_DIR / "PokemonAPI"
OUTPUT_DIR      = DATASET_DIR / "Synthetic"
ROSTERS_DIR     = Path(__file__).parent / "Pokemon Snap Rosters"


IMAGE_W, IMAGE_H    = 1920, 1080
NUM_IMAGES          = 10000
FIRST_SPAWN_CHANCE  = 0.15
ROTATION_CHANCE     = 0.35
COLOR_SWAP_CHANCE   = 0.25
MIN_SPRITE_SCALE    = 0.15
MAX_SPRITE_SCALE    = 0.60



def load_sprites(pokemon_set: set[str]) -> dict[str, list[Path]]:
    """Load all valid (non-3D) sprite paths for each Pokemon in the set."""
    sprites = {}
    for name in pokemon_set:
        folder = SPRITES_DIR / name
        if not folder.exists():
            folder = SPRITES_DIR / name.capitalize()
        if not folder.exists():
            print(f"Warning: no sprite folder for {name}, skipping.")
            continue
        valid_paths = []
        for p in folder.rglob("*.png"):
            if _is_3d(p.name):
                continue
            try:
                with Image.open(p) as img:
                    img.verify()
                valid_paths.append(p)
            except Exception:
                print(f"Warning: skipping unreadable sprite {p.name}")
        paths = valid_paths
        if paths:
            sprites[name] = paths
    return sprites


def pick_pokemon_for_image(pokemon_names: list[str]) -> list[str]:
    appearances = []
    for name in pokemon_names:
        chance = FIRST_SPAWN_CHANCE
        while random.random() < chance:
            appearances.append(name)
            chance /= 2
    return appearances


def rotate_sprite(img: Image.Image) -> Image.Image:
    angle = random.uniform(0, 360)
    return img.rotate(angle, expand=True, resample=Image.BICUBIC)


def color_swap_sprite(img: Image.Image) -> Image.Image:
    """
    Shifts the hue of all colored pixels by a random amount.
    """
    has_alpha = img.mode == "RGBA"
    if has_alpha:
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
    else:
        rgb = img.convert("RGB")

    arr = np.array(rgb, dtype=np.float32) / 255.0
    max_c = arr.max(axis=2)
    min_c = arr.min(axis=2)
    delta = max_c - min_c

    h = np.zeros_like(max_c)
    s = np.where(max_c > 0, delta / max_c, 0.0)
    v = max_c

    r_arr, g_arr, b_arr = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    mask_r = (max_c == r_arr) & (delta > 0)
    mask_g = (max_c == g_arr) & (delta > 0)
    mask_b = (max_c == b_arr) & (delta > 0)
    h[mask_r] = (60 * ((g_arr - b_arr) / delta % 6))[mask_r]
    h[mask_g] = (60 * ((b_arr - r_arr) / delta + 2))[mask_g]
    h[mask_b] = (60 * ((r_arr - g_arr) / delta + 4))[mask_b]

    # Shift hue by random amount, keep saturation/value untouched
    shift = random.uniform(30, 330)
    h = (h + shift) % 360

    # HSV -> RGB
    hi = (h / 60).astype(int) % 6
    f  = (h / 60) - np.floor(h / 60)
    p  = v * (1 - s)
    q  = v * (1 - f * s)
    t  = v * (1 - (1 - f) * s)

    out = np.zeros_like(arr)
    for val, mask in [
        (np.stack([v, t, p], axis=2), hi == 0),
        (np.stack([q, v, p], axis=2), hi == 1),
        (np.stack([p, v, t], axis=2), hi == 2),
        (np.stack([p, q, v], axis=2), hi == 3),
        (np.stack([t, p, v], axis=2), hi == 4),
        (np.stack([v, p, q], axis=2), hi == 5),
    ]:
        out[mask] = val[mask]

    result = Image.fromarray((out * 255).clip(0, 255).astype(np.uint8), "RGB")

    if has_alpha:
        result = Image.merge("RGBA", (*result.split(), a))

    return result


def compute_bbox(x: int, y: int, w: int, h: int) -> tuple[float, float, float, float]:
    cx = (x + w / 2) / IMAGE_W
    cy = (y + h / 2) / IMAGE_H
    nw = w / IMAGE_W
    nh = h / IMAGE_H

    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def place_sprite(background: Image.Image, sprite: Image.Image, class_id: int) -> tuple[Image.Image, tuple[int, float, float, float, float]]:
    """
    Scale, optionally rotate, and paste a sprite at a random position.
    Returns updated background and the YOLO label tuple.
    """
    scale = random.uniform(MIN_SPRITE_SCALE, MAX_SPRITE_SCALE)
    new_h = int(IMAGE_H * scale)
    ratio = new_h / sprite.height
    new_w = int(sprite.width * ratio)
    sprite = sprite.resize((new_w, new_h), Image.LANCZOS)

    if random.random() < ROTATION_CHANCE:
        sprite = rotate_sprite(sprite)

    if random.random() < COLOR_SWAP_CHANCE:
        sprite = color_swap_sprite(sprite)

    sw, sh = sprite.size

    max_x = max(0, IMAGE_W - sw)
    max_y = max(0, IMAGE_H - sh)
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    if sprite.mode != "RGBA":
        sprite = sprite.convert("RGBA")
    background.paste(sprite, (x, y), mask=sprite.split()[3])

    label = (class_id, *compute_bbox(x, y, sw, sh))
    return background, label


def generate(levels=None, num_images: int = NUM_IMAGES) -> None:
    """
    Generate synthetic YOLO training images.

    :param levels: Level list for load_pokemon_for_levels, defaults to [1] (Beach).
    :param num_images: Number of images to generate.
    """
    if levels is None:
        levels = [1]

    pokemon_set = load_pokemon_for_levels(levels)
    pokemon_names = sorted(pokemon_set)
    class_map = {name: i for i, name in enumerate(pokemon_names)}

    print(f"Roster ({len(pokemon_names)} Pokemon): {pokemon_names}")

    # Load sprites
    sprites = load_sprites(pokemon_set)
    available = [n for n in pokemon_names if n in sprites]
    bg_paths = list(BACKGROUNDS_DIR.glob("*.png")) + list(BACKGROUNDS_DIR.glob("*.jpg"))

    img_out   = OUTPUT_DIR / "images"
    label_out = OUTPUT_DIR / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    empty_count = 0

    for i in tqdm(range(num_images), desc="Generating synthetic images"):
        bg_path = random.choice(bg_paths)
        bg = Image.open(bg_path).convert("RGB").resize((IMAGE_W, IMAGE_H), Image.LANCZOS)
        appearances = pick_pokemon_for_image(available)
        random.shuffle(appearances)

        labels = []
        for name in appearances:
            sprite_path = random.choice(sprites[name])
            try:
                sprite = Image.open(sprite_path).convert("RGBA")
            except Exception:
                continue
            class_id = class_map[name]
            bg, label = place_sprite(bg, sprite, class_id)
            labels.append(label)

        if not labels:
            empty_count += 1

        img_name = f"synthetic_{i:05d}.jpg"
        bg.save(img_out / img_name, quality=95)

        label_name = f"synthetic_{i:05d}.txt"
        with open(label_out / label_name, "w") as f:
            for label in labels:
                class_id, cx, cy, w, h = label
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"train: {img_out.resolve()}\n")
        f.write(f"val: {img_out.resolve()}\n\n")
        f.write(f"nc: {len(pokemon_names)}\n")
        f.write(f"names: {pokemon_names}\n")


if __name__ == "__main__":
    generate(levels=[1], num_images=NUM_IMAGES)
