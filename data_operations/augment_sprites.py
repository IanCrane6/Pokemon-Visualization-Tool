import io
import random
from pathlib import Path

import numpy as np
from PIL import (Image, ImageFilter, ImageEnhance, ImageOps,
                 ImageDraw)
from tqdm import tqdm

from data_selection import select_data

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "Dataset" / "AugmentedSprites"
BACKGROUNDS_DIR = BASE_DIR / "Dataset" / "Backgrounds"
IMAGES_PER_POKEMON = 200

np.seterr(divide='ignore', invalid='ignore')

def aug_rotate(img: Image.Image) -> Image.Image:
    angle = random.uniform(0, 360)
    return img.rotate(angle, expand=True, resample=Image.BICUBIC)


def aug_flip(img: Image.Image) -> Image.Image:
    op = random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM])
    return img.transpose(op)


def aug_hue_swap(img: Image.Image) -> Image.Image:
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

    r_a, g_a, b_a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mask_r = (max_c == r_a) & (delta > 0)
    mask_g = (max_c == g_a) & (delta > 0)
    mask_b = (max_c == b_a) & (delta > 0)
    h[mask_r] = (60 * ((g_a - b_a) / delta % 6))[mask_r]
    h[mask_g] = (60 * ((b_a - r_a) / delta + 2))[mask_g]
    h[mask_b] = (60 * ((r_a - g_a) / delta + 4))[mask_b]

    shift = random.uniform(30, 330)
    h = (h + shift) % 360

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


def aug_fade(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    factor = random.uniform(0.4, 0.85)
    r, g, b, a = img.split()
    a = a.point(lambda x: int(x * factor))
    return Image.merge("RGBA", (r, g, b, a))


def aug_blur(img: Image.Image) -> Image.Image:
    radius = random.uniform(0.5, 3.0)
    has_alpha = img.mode == "RGBA"
    if has_alpha:
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b)).filter(ImageFilter.GaussianBlur(radius))
        return Image.merge("RGBA", (*rgb.split(), a))
    return img.filter(ImageFilter.GaussianBlur(radius))


def aug_jpeg_noise(img: Image.Image) -> Image.Image:
    has_alpha = img.mode == "RGBA"
    if has_alpha:
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
    else:
        rgb = img.convert("RGB")

    quality = random.randint(30, 75)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    rgb = Image.open(buf).copy()

    if has_alpha:
        return Image.merge("RGBA", (*rgb.split(), a))
    return rgb


def aug_gaussian_noise(img: Image.Image) -> Image.Image:
    has_alpha = img.mode == "RGBA"
    if has_alpha:
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
    else:
        rgb = img.convert("RGB")

    arr = np.array(rgb, dtype=np.float32)
    noise = np.random.normal(0, random.uniform(5, 25), arr.shape)
    arr = (arr + noise).clip(0, 255).astype(np.uint8)
    rgb = Image.fromarray(arr)

    if has_alpha:
        return Image.merge("RGBA", (*rgb.split(), a))
    return rgb


def aug_color_jitter(img: Image.Image) -> Image.Image:
    has_alpha = img.mode == "RGBA"
    if has_alpha:
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
    else:
        rgb = img.convert("RGB")

    rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(0.5, 1.5))
    rgb = ImageEnhance.Contrast(rgb).enhance(random.uniform(0.5, 1.5))
    rgb = ImageEnhance.Color(rgb).enhance(random.uniform(0.4, 1.6))

    if has_alpha:
        return Image.merge("RGBA", (*rgb.split(), a))
    return rgb


def aug_sharpen(img: Image.Image) -> Image.Image:
    has_alpha = img.mode == "RGBA"
    if has_alpha:
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
    else:
        rgb = img.convert("RGB")

    op = random.choice([ImageFilter.SHARPEN, ImageFilter.EDGE_ENHANCE,
                        ImageFilter.SMOOTH, ImageFilter.DETAIL])
    rgb = rgb.filter(op)

    if has_alpha:
        return Image.merge("RGBA", (*rgb.split(), a))
    return rgb


def aug_perspective(img: Image.Image) -> Image.Image:
    w, h = img.size
    margin = int(min(w, h) * 0.15)

    def jitter():
        return random.randint(-margin, margin)

    coeffs_src = [(0, 0), (w, 0), (w, h), (0, h)]
    coeffs_dst = [
                    (jitter(), jitter()),
                    (w + jitter(), jitter()),
                    (w + jitter(), h + jitter()),
                    (jitter(), h + jitter()),
                 ]

    def find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix += [
                        [p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]],
                        [0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]]
                      ]
        A = np.array(matrix, dtype=np.float64)
        B = np.array([x for p in pb for x in p], dtype=np.float64)
        res = np.linalg.lstsq(A, B, rcond=None)[0]
        return np.append(res, 1).tolist()[:8]

    try:
        coeffs = find_coeffs(coeffs_dst, coeffs_src)
        return img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    except Exception:
        return img


def aug_grayscale(img: Image.Image) -> Image.Image:
    has_alpha = img.mode == "RGBA"
    if has_alpha:
        r, g, b, a = img.split()
        rgb = ImageOps.grayscale(Image.merge("RGB", (r, g, b))).convert("RGB")
        return Image.merge("RGBA", (*rgb.split(), a))
    return ImageOps.grayscale(img).convert("RGB")


def aug_cutout(img: Image.Image) -> Image.Image:
    img = img.copy()
    w, h = img.size
    cw = random.randint(w // 8, w // 3)
    ch = random.randint(h // 8, h // 3)
    cx = random.randint(0, w - cw)
    cy = random.randint(0, h - ch)
    draw = ImageDraw.Draw(img)
    draw.rectangle([cx, cy, cx + cw, cy + ch], fill=(0, 0, 0, 0) if img.mode == "RGBA" else (0, 0, 0))
    return img


def composite_on_background(sprite: Image.Image, bg_paths: list[Path]) -> Image.Image:
    bg = Image.open(random.choice(bg_paths)).convert("RGB")
    sw, sh = sprite.size
    bw, bh = bg.size
    crop_size = min(bw, bh, sw * 2, sh * 2)
    cx = random.randint(0, max(0, bw - crop_size))
    cy = random.randint(0, max(0, bh - crop_size))
    bg = bg.crop((cx, cy, cx + crop_size, cy + crop_size)).resize((sw, sh), Image.LANCZOS)

    if sprite.mode == "RGBA":
        bg.paste(sprite, (0, 0), mask=sprite.split()[3])
    else:
        bg.paste(sprite.convert("RGBA"), (0, 0))
    return bg


# Pool of all augmentations with their probability of being applied each time
AUGMENTATION_POOL = [
    (aug_rotate,        0.7),
    (aug_flip,          0.5),
    (aug_hue_swap,      0.4),
    (aug_fade,          0.3),
    (aug_blur,          0.5),
    (aug_jpeg_noise,    0.5),
    (aug_gaussian_noise,0.4),
    (aug_color_jitter,  0.7),
    (aug_sharpen,       0.4),
    (aug_perspective,   0.3),
    (aug_grayscale,     0.1),
    (aug_cutout,        0.3),
]


def apply_random_augmentations(img: Image.Image) -> Image.Image:
    for fn, prob in AUGMENTATION_POOL:
        if random.random() < prob:
            try:
                img = fn(img)
            except Exception:
                pass
    return img


def verify(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def generate_augmented_dataset(images_per_pokemon: int = IMAGES_PER_POKEMON) -> None:
    df = select_data(sources=['PokemonAPI'], levels=[1], exclude_3d=True, return_3d=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bg_paths = list(BACKGROUNDS_DIR.glob("*.png")) + list(BACKGROUNDS_DIR.glob("*.jpg"))
    total_written = 0

    for pokemon_name, group in tqdm(df.groupby("label"), desc="Pokemon"):
        out_folder = OUTPUT_DIR / pokemon_name.lower()
        out_folder.mkdir(parents=True, exist_ok=True)

        valid_sprites = []
        for img_path in group["image_path"]:
            img_path = Path(img_path)
            if not verify(img_path):
                continue
            try:
                img = Image.open(img_path).copy()
                valid_sprites.append((img_path.stem, img))
            except Exception:
                continue

        if not valid_sprites:
            continue

        written = 0
        idx = 0

        while written < images_per_pokemon:
            stem, original = valid_sprites[idx % len(valid_sprites)]
            idx += 1

            try:
                aug = apply_random_augmentations(original.copy())
                if bg_paths and random.random() < 0.4:
                    aug = composite_on_background(aug, bg_paths)

                aug.save(out_folder / f"{stem}_{written:04d}.png")
                written += 1
                total_written += 1
            except Exception as e:
                print(f"Warning: failed for {pokemon_name}/{stem}: {e}")

    print(f"\nDone. {total_written} total images written to {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_augmented_dataset()
