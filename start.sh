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

# 3) basicsr torchvision uyumluluk patch'i
DEGRADATIONS_FILE="./venv/lib/python3.*/site-packages/basicsr/data/degradations.py"
for f in $DEGRADATIONS_FILE; do
    if [ -f "$f" ] && grep -q "functional_tensor" "$f" 2>/dev/null; then
        sed -i.bak 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$f"
        echo "      ✓ basicsr uyumluluk patch'i uygulandı"
    fi
done

# 4) Başlat
echo "[3/3] Uygulama başlatılıyor..."
echo ""
echo "  ➜  http://localhost:5001"
echo ""
./venv/bin/python app.py
