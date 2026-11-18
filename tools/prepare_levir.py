"""Prepare the 256-by-256 LEVIR-CD tiles listed in the release manifests."""

import argparse
from pathlib import Path

from PIL import Image

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("dataset/ChangeDetection/LEVIR"))
    parser.add_argument("--manifests", type=Path, default=Path("splits/LEVIR-CD"))
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Source and output must be separate directories")

    for split in ("train", "val", "test"):
        entries = (args.manifests / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        groups = {}
        for filename in entries:
            if not filename:
                continue
            stem, index = Path(filename).stem.rsplit("_", 1)
            groups.setdefault(stem, []).append((filename, int(index)))
        for channel in ("A", "B", "label"):
            dest = args.output / split / channel
            dest.mkdir(parents=True, exist_ok=True)
            for stem, patches in groups.items():
                with Image.open(args.source / split / channel / f"{stem}.png") as image:
                    if image.size != (1024, 1024):
                        raise ValueError(f"Expected 1024x1024 pixels: {image.filename}")
                    for filename, index in patches:
                        if not 1 <= index <= 16:
                            raise ValueError(f"Invalid tile index: {filename}")
                        row, col = divmod(index - 1, 4)
                        x, y = col * 256, row * 256
                        tile = image.crop((x, y, x + 256, y + 256))
                        if channel == "label":
                            tile = tile.convert("L").point(lambda value: 255 if value else 0)
                        tile.save(dest / filename)
        print(f"{split}: {sum(map(len, groups.values()))} paired tiles")

if __name__ == "__main__":
    main()
