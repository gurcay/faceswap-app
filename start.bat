@echo off
chcp 65001 >nul 2>&1
title FaceSwap - AI Yüz Değiştirme
cd /d "%~dp0"

echo ============================================================
echo   FaceSwap - AI Yüz Değiştirme Uygulaması
echo ============================================================
echo.

:: 1) Virtual environment
if not exist "venv\Scripts\python.exe" (
    echo [1/3] Virtual environment oluşturuluyor...
    python -m venv venv
    if errorlevel 1 (
        echo HATA: Python bulunamadı. python.org'dan Python 3.8+ kurun.
        pause
        exit /b 1
    )
    venv\Scripts\pip install --upgrade pip -q
    echo       Bağımlılıklar kuruluyor ^(bu biraz sürebilir^)...
    venv\Scripts\pip install -r requirements.txt -q
    echo       Kurulum tamamlandı.
) else (
    echo [1/3] Virtual environment mevcut.
)

:: 2) Modeller
if not exist "models\inswapper_128.onnx" (
    echo [2/3] AI modelleri indiriliyor ^(ilk seferde ~900 MB^)...
    venv\Scripts\python download_model.py
) else if not exist "models\GFPGANv1.4.pth" (
    echo [2/3] AI modelleri indiriliyor...
    venv\Scripts\python download_model.py
) else (
    echo [2/3] AI modelleri mevcut.
)

:: 3) basicsr/torchvision uyumluluk patch
venv\Scripts\python -c "
import pathlib, re
for f in pathlib.Path('venv/Lib/site-packages/basicsr/data').glob('degradations.py'):
    txt = f.read_text()
    if 'functional_tensor' in txt:
        f.write_text(txt.replace('from torchvision.transforms.functional_tensor import rgb_to_grayscale', 'from torchvision.transforms.functional import rgb_to_grayscale'))
        print('      basicsr uyumluluk patchi uygulandi.')
" 2>nul

:: 4) Başlat
echo [3/3] Uygulama başlatılıyor...
echo.
echo   http://localhost:5001
echo.
venv\Scripts\python app.py
pause
