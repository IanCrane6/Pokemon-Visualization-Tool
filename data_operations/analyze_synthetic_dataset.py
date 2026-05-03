from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt


BASE_DIR   = Path(__file__).parent.parent
LABEL_DIR  = BASE_DIR / "Dataset" / "Synthetic" / "labels"
NAMES_FILE = BASE_DIR / "Dataset" / "Synthetic" / "data.yaml"


def load_class_names() -> list[str]:
    if NAMES_FILE.exists():
        with open(NAMES_FILE) as f:
            for line in f:
                if line.startswith("names:"):
                    names_str = line.split(":", 1)[1].strip()
                    names = [n.strip().strip("'\"[]") for n in names_str.split(",")]
                    return names
    return None


def analyze() -> None:
    label_files = list(LABEL_DIR.glob("*.txt"))


    class_counter  = Counter()
    sprites_per_image = Counter()

    for label_file in label_files:
        with open(label_file) as f:
            lines = [l.strip() for l in f if l.strip()]

        sprites_per_image[len(lines)] += 1

        for line in lines:
            class_id = int(line.split()[0])
            class_counter[class_id] += 1

    class_names = load_class_names()

    sorted_classes = sorted(class_counter.keys())
    labels = [class_names[c] if class_names and c < len(class_names)
              else str(c) for c in sorted_classes]
    counts = [class_counter[c] for c in sorted_classes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    bars = ax1.bar(labels, counts, color="steelblue", edgecolor="white")
    ax1.set_title("Total Pokemon Appearances in Synthetic Dataset", fontsize=13)
    ax1.set_xlabel("Pokemon Class")
    ax1.set_ylabel("Total Appearances")
    ax1.tick_params(axis="x", rotation=45)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(counts) * 0.01,
                 str(count), ha="center", va="bottom", fontsize=8)

    sprite_counts = sorted(sprites_per_image.keys())
    image_counts  = [sprites_per_image[s] for s in sprite_counts]
    x_labels = [str(s) for s in sprite_counts]

    bars2 = ax2.bar(x_labels, image_counts, color="darkorange", edgecolor="white")
    ax2.set_title("Number of Sprites Per Image", fontsize=13)
    ax2.set_xlabel("Sprites in Image")
    ax2.set_ylabel("Number of Images")
    for bar, count in zip(bars2, image_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(image_counts) * 0.01,
                 str(count), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = BASE_DIR / "Dataset" / "Synthetic" / "dataset_analysis.png"
    plt.savefig(out_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    analyze()
