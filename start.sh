#!/bin/bash
# FaceSwap - Tek komutla kurulum ve başlatma

set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  FaceSwap - AI Yüz Değiştirme Uygulaması"
echo "============================================================"
echo ""

# 1) Virtual environment
if [ ! -f "venv/bin/python" ]; then
    echo "[1/3] Virtual environment oluşturuluyor..."
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip -q
    echo "      Bağımlılıklar kuruluyor (bu biraz sürebilir)..."
    ./venv/bin/pip install -r requirements.txt -q
    echo "      ✓ Kurulum tamamlandı"
else
    echo "[1/3] ✓ Virtual environment mevcut"
fi

# 2) Modeller
if [ ! -f "models/inswapper_128.onnx" ] || [ ! -f "models/GFPGANv1.4.pth" ]; then
    echo "[2/3] AI modelleri indiriliyor (ilk seferde ~900 MB)..."
    ./venv/bin/python download_model.py
else
    echo "[2/3] ✓ AI modelleri mevcut"
fi

# 3) basicsr torchvision uyumluluk patch'i (cross-platform Python ile)
./venv/bin/python -c "
import pathlib
for f in pathlib.Path('venv').rglob('basicsr/data/degradations.py'):
    txt = f.read_text()
    if 'functional_tensor' in txt:
        f.write_text(txt.replace('from torchvision.transforms.functional_tensor import rgb_to_grayscale', 'from torchvision.transforms.functional import rgb_to_grayscale'))
        print('      ✓ basicsr uyumluluk patch\'i uygulandı')
" 2>/dev/null

# 4) Başlat
echo "[3/3] Uygulama başlatılıyor..."
echo ""
echo "  ➜  http://localhost:5001"
echo ""
./venv/bin/python app.py
