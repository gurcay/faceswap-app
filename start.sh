#!/bin/bash
# FaceSwap - Tek komutla kurulum ve başlatma

set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  FaceSwap - AI Yüz Değiştirme Uygulaması"
echo "============================================================"
echo ""

# 0) Python kontrol - yoksa otomatik kur
if ! command -v python3 &>/dev/null; then
    echo "[0/4] Python bulunamadı. Otomatik kuruluyor..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS: Homebrew ile
        if ! command -v brew &>/dev/null; then
            echo "      Homebrew kuruluyor..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        echo "      Python kuruluyor (brew)..."
        brew install python@3.11
    else
        # Linux (Debian/Ubuntu)
        echo "      Python kuruluyor (apt)..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3 python3-venv python3-pip
    fi

    if ! command -v python3 &>/dev/null; then
        echo "      ✗ Python kurulamadı. Manuel kurun: https://www.python.org/downloads/"
        exit 1
    fi
    echo "      ✓ Python kuruldu"
else
    echo "[0/4] ✓ Python mevcut"
fi
python3 --version

# 1) Virtual environment
if [ ! -f "venv/bin/python" ]; then
    echo "[1/4] Virtual environment oluşturuluyor..."
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip -q
    echo "      Bağımlılıklar kuruluyor (bu biraz sürebilir)..."
    ./venv/bin/pip install -r requirements.txt -q
    echo "      ✓ Kurulum tamamlandı"
else
    echo "[1/4] ✓ Virtual environment mevcut"
fi

# 2) Modeller
if [ ! -f "models/inswapper_128.onnx" ] || [ ! -f "models/GFPGANv1.4.pth" ]; then
    echo "[2/4] AI modelleri indiriliyor (ilk seferde ~900 MB)..."
    ./venv/bin/python download_model.py
else
    echo "[2/4] ✓ AI modelleri mevcut"
fi

# 3) basicsr torchvision uyumluluk patch'i
./venv/bin/python -c "
import pathlib
for f in pathlib.Path('venv').rglob('basicsr/data/degradations.py'):
    txt = f.read_text()
    if 'functional_tensor' in txt:
        f.write_text(txt.replace('from torchvision.transforms.functional_tensor import rgb_to_grayscale', 'from torchvision.transforms.functional import rgb_to_grayscale'))
        print('      ✓ basicsr uyumluluk patch\'i uygulandı')
" 2>/dev/null

# 4) Başlat
echo "[3/4] Uygulama başlatılıyor..."
echo ""
echo "  ➜  http://localhost:5001"
echo ""
./venv/bin/python app.py
