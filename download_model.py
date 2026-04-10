#!/usr/bin/env python3
"""
FaceSwap için gerekli tüm AI modellerini indir.
- inswapper_128.onnx (~530 MB) - Face swap modeli
- GFPGANv1.4.pth (~332 MB) - Yüz iyileştirme modeli
- buffalo_l (~280 MB) - Yüz algılama modeli (InsightFace, otomatik indirilir)
"""

import sys
from pathlib import Path


MODELS = [
    {
        'name': 'inswapper_128.onnx',
        'path': Path('models/inswapper_128.onnx'),
        'min_size': 100_000_000,
        'urls': [
            'https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx',
            'https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx',
        ],
        'desc': 'Face swap modeli (~530 MB)'
    },
    {
        'name': 'GFPGANv1.4.pth',
        'path': Path('models/GFPGANv1.4.pth'),
        'min_size': 200_000_000,
        'urls': [
            'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
        ],
        'desc': 'Yüz iyileştirme modeli (~332 MB)'
    },
]


def download_with_progress(url, dest_path, desc):
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("Hata: requests ve tqdm gerekli. Önce: pip install requests tqdm")
        sys.exit(1)

    print(f"  İndiriliyor: {url}")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total = int(response.headers.get('content-length', 0))
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = dest_path.with_suffix('.tmp')
    with open(tmp_path, 'wb') as f, tqdm(
        total=total, unit='iB', unit_scale=True, unit_divisor=1024, desc=desc
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

    tmp_path.rename(dest_path)


def download_model(model):
    path = model['path']

    if path.exists() and path.stat().st_size > model['min_size']:
        print(f"  ✓ Zaten mevcut ({path.stat().st_size / 1024 / 1024:.0f} MB)")
        return True

    if path.exists():
        path.unlink()

    for i, url in enumerate(model['urls']):
        try:
            print(f"  Kaynak {i+1}/{len(model['urls'])} deneniyor...")
            download_with_progress(url, path, model['name'])
            if path.exists() and path.stat().st_size > model['min_size']:
                print(f"  ✓ Başarılı ({path.stat().st_size / 1024 / 1024:.0f} MB)")
                return True
        except Exception as e:
            print(f"  ✗ Başarısız: {e}")
            if path.with_suffix('.tmp').exists():
                path.with_suffix('.tmp').unlink()

    print(f"  ✗ İndirilemedi! Manuel indirin ve {path.absolute()} konumuna koyun.")
    return False


def main():
    Path('models').mkdir(exist_ok=True)

    print("=" * 60)
    print("  FaceSwap - Model İndirici")
    print("=" * 60)
    print("\n  Bu modeller eğlence amaçlı kullanım içindir.")
    print("  Başkasının izni olmadan yüz swap yapmayın.\n")

    success = 0
    for model in MODELS:
        print(f"\n[{model['name']}] {model['desc']}")
        if download_model(model):
            success += 1

    print(f"\n{'=' * 60}")
    print(f"  {success}/{len(MODELS)} model hazır.")

    if success == len(MODELS):
        print("  Uygulamayı başlatın: ./start.sh")
    else:
        print("  Bazı modeller indirilemedi. Tekrar deneyin veya manuel indirin.")
        sys.exit(1)


if __name__ == '__main__':
    main()
