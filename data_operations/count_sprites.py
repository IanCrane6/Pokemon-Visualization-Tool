from pathlib import Path
import matplotlib.pyplot as plt
from data_selection import _is_3d, load_pokemon_for_levels

BASE_DIR = Path(__file__).parent.parent
SPRITES_DIR = BASE_DIR / "Dataset" / "PokemonAPI"

def count_sprites() -> dict[str, int]:
    pokemon_set = load_pokemon_for_levels([1])
    counts = {}

    for name in sorted(pokemon_set):
        folder = SPRITES_DIR / name
        if not folder.exists():
            folder = SPRITES_DIR / name.capitalize()
        valid = [p for p in folder.rglob("*.png") if not _is_3d(p.name)]
        counts[name] = len(valid)

    names  = list(counts.keys())
    values = [counts[n] for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, values, color="steelblue", edgecolor="white")
    ax.set_title("Non-3D Sprites per Level 1 Pokemon (PokemonAPI)", fontsize=13)
    ax.set_xlabel("Pokemon")
    ax.set_ylabel("Sprite Count")
    ax.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = BASE_DIR / "Dataset" / "sprite_counts.png"
    plt.savefig(out_path, dpi=150)
    print(f"Chart saved to {out_path}")
    plt.show()
    return counts


if __name__ == "__main__":
    count_sprites()
