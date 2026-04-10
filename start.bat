@echo off
chcp 65001 >nul 2>&1
title FaceSwap - AI Yüz Değiştirme
cd /d "%~dp0"

echo ============================================================
echo   FaceSwap - AI Yüz Değiştirme Uygulaması
echo ============================================================
echo.

:: 0) Python kontrol - yoksa otomatik kur
where python >nul 2>&1
if errorlevel 1 (
    echo [0/4] Python bulunamadı. Otomatik kuruluyor...
    echo       Python 3.11.9 indiriliyor...

    :: Python installer indir
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile '%TEMP%\python_installer.exe'}"

    if not exist "%TEMP%\python_installer.exe" (
        echo       HATA: Python indirilemedi. Lütfen manuel kurun: https://www.python.org/downloads/
        pause
        exit /b 1
    )

    echo       Python kuruluyor (bu birkaç dakika sürebilir)...
    :: Sessiz kurulum: PATH'e ekle, pip dahil, tüm kullanıcılar için
    "%TEMP%\python_installer.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_launcher=1

    if errorlevel 1 (
        echo       Sessiz kurulum başarısız. Yönetici izniyle deniyor...
        powershell -Command "Start-Process '%TEMP%\python_installer.exe' -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1 Include_pip=1' -Verb RunAs -Wait"
    )

    :: PATH'i yenile (yeni terminalde geçerli olur, burada manuel ekle)
    set "PATH=%LocalAppData%\Programs\Python\Python311;%LocalAppData%\Programs\Python\Python311\Scripts;%ProgramFiles%\Python311;%ProgramFiles%\Python311\Scripts;%PATH%"

    :: Kontrol
    where python >nul 2>&1
    if errorlevel 1 (
        echo       HATA: Python kurulumu tamamlanamadı.
        echo       Lütfen bilgisayarı yeniden başlatın ve start.bat'ı tekrar çalıştırın.
        echo       Veya manuel kurun: https://www.python.org/downloads/
        pause
        exit /b 1
    )

    echo       Python kuruldu.
    del "%TEMP%\python_installer.exe" >nul 2>&1
) else (
    echo [0/4] Python mevcut.
)

:: Python versiyonunu göster
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo       %%i

:: 1) Virtual environment
if not exist "venv\Scripts\python.exe" (
    echo [1/4] Virtual environment oluşturuluyor...
    python -m venv venv
    if errorlevel 1 (
        echo HATA: Virtual environment oluşturulamadı.
        pause
        exit /b 1
    )
    venv\Scripts\pip install --upgrade pip -q
    echo       Bağımlılıklar kuruluyor ^(bu biraz sürebilir^)...
    venv\Scripts\pip install -r requirements.txt -q
    echo       Kurulum tamamlandı.
) else (
    echo [1/4] Virtual environment mevcut.
)

:: 2) Modeller
if not exist "models\inswapper_128.onnx" (
    echo [2/4] AI modelleri indiriliyor ^(ilk seferde ~900 MB^)...
    venv\Scripts\python download_model.py
) else if not exist "models\GFPGANv1.4.pth" (
    echo [2/4] AI modelleri indiriliyor...
    venv\Scripts\python download_model.py
) else (
    echo [2/4] AI modelleri mevcut.
)

:: 3) basicsr/torchvision uyumluluk patch
venv\Scripts\python -c "import pathlib; [f.write_text(f.read_text().replace('from torchvision.transforms.functional_tensor import rgb_to_grayscale','from torchvision.transforms.functional import rgb_to_grayscale')) or print('      basicsr patchi uygulandi.') for f in pathlib.Path('venv').rglob('basicsr/data/degradations.py') if 'functional_tensor' in f.read_text()]" 2>nul

:: 4) Başlat
echo [3/4] Uygulama başlatılıyor...
echo.
echo   http://localhost:5001
echo.
venv\Scripts\python app.py
pause
