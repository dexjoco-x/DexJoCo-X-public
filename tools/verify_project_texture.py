#!/usr/bin/env python3
"""Verify the historical DIG texture recipe in memory; never edit the texture."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((ROOT / 'docs/PROJECT_ASSET_PROVENANCE.json').read_text())
    if hashlib.sha256(args.font.read_bytes()).hexdigest() != receipt['font_sha256']:
        raise ValueError('Use the exact historical font for this verification')
    expected = receipt['files'][0]
    path = ROOT / expected['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
        raise ValueError('Original texture changed')
    rendered = Image.new('RGB', (4000, 4000), (0, 0, 0))
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.truetype(str(args.font), 1680)
    bbox = draw.textbbox((0, 0), 'DIG', font=font)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    xy = (int((4000 - width) * 0.48 - bbox[0]), int(4000 * 0.53 - height / 2 - bbox[1]))
    draw.text(xy, 'DIG', font=font, fill=(255, 255, 255))
    with Image.open(path) as image:
        original = image.convert('RGB').tobytes()
    if original != rendered.tobytes() or hashlib.sha256(original).hexdigest() != expected['pixel_sha256']:
        raise ValueError('Historical recipe did not reproduce the complete pixel buffer')
    print(json.dumps({'passed': True, 'pixel_exact': True, 'asset_modified': False}))


if __name__ == '__main__':
    main()
