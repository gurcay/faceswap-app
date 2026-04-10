# FaceSwap App

Lokal ağda (on-premise) çalışan, çocuklara yönelik AI tabanlı yüz değiştirme (face swap) web uygulaması. Etkinliklerde, festivallerde veya eğitim ortamlarında çocukların farklı mesleklerde kendilerini görmesini sağlar.

Çocuk bir meslek ve şablon seçer, webcam otomatik olarak fotoğrafını çeker, AI yüz değiştirme işlemini yapar ve sonucu QR kodu ile telefonuna gönderir. Tüm süreç tek tıklamayla (şablon seçimi) başlar ve kullanıcı müdahalesi gerektirmeden tamamlanır.

## Mimari Genel Bakış

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                              │
│  Tailwind CSS + Vanilla JS (tek HTML dosyası)               │
│                                                              │
│  Wizard Akışı:                                               │
│  [Meslek Seç] → [Cinsiyet Seç] → [Şablon Seç] → [Otomatik] │
│                                                              │
│  Şablon seçildiğinde:                                        │
│  1. Webcam açılır (getUserMedia)                             │
│  2. 5 sn geri sayım                                         │
│  3. Otomatik çekim (canvas)                                  │
│  4. Base64 olarak backend'e POST                             │
│  5. Sonuç + QR kodu gösterilir                               │
└──────────────────────┬──────────────────────────────────────┘
                       │ POST /api/capture
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                        BACKEND (Flask)                       │
│                                                              │
│  Face Swap Pipeline (3 aşamalı):                             │
│                                                              │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ InsightFace  │──▶│ inswapper_128│──▶│ GFPGAN + seamless│  │
│  │ (detection + │   │ (face swap)  │   │ (face restore +  │  │
│  │  recognition)│   │              │   │  natural blend)  │  │
│  └─────────────┘   └──────────────┘   └──────────────────┘  │
│                                                              │
│  Sonuç: JPEG → tmpfiles.org'a yükle → download URL döndür   │
└─────────────────────────────────────────────────────────────┘
```

## AI Pipeline Detayı

Yüz değiştirme tek bir model çağrısı değil, 3 aşamalı bir pipeline olarak çalışır. Her aşama bir öncekinin çıktısını alır:

### Aşama 1: Yüz Algılama ve Tanıma (InsightFace buffalo_l)

```python
face_analyzer = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
```

- **Ne yapar:** Hem kaynak (webcam) hem hedef (şablon) görüntüde yüzleri bulur
- **Çıktı:** Bounding box koordinatları, 5 noktalı landmark, 512 boyutlu ArcFace embedding
- **Neden `buffalo_l`:** En doğru açık kaynak yüz algılama modeli. `detection` modülü yüz konumunu, `recognition` modülü yüz kimliğini (embedding) verir
- **Neden `allowed_modules` kısıtlaması:** Gereksiz modülleri (landmark_3d, genderage) yüklemeyerek bellek ve başlatma süresinden tasarruf
- **Kaynak boyut:** ~280 MB (ilk çalıştırmada otomatik indirilir)

### Aşama 2: Yüz Değiştirme (inswapper_128)

```python
result = face_swapper.get(result, target_face, source_face, paste_back=True)
```

- **Ne yapar:** Kaynak yüzün kimliğini (ArcFace embedding) hedef yüzün pozisyonuna, açısına ve ifadesine uyarlayarak yeni bir yüz üretir
- **Çalışma prensibi:** Kaynak yüzün 512-d embedding vektörünü alır, hedef yüzün hizalanmış 128x128 görüntüsünü girdi olarak kullanır, çıktı olarak kaynak kimliğiyle hedef pozu birleştiren yeni bir 128x128 yüz üretir, bunu orijinal görüntüye geri yapıştırır
- **Kısıtlama:** 128x128 çözünürlük — büyük yüzlerde pikselleşme yapar
- **Neden bu model:** deepinsight/insightface tarafından geliştirilen en yaygın ve güvenilir açık kaynak face swap modeli. `paste_back=True` parametresi affine transform ile otomatik geri yapıştırma yapar
- **Model boyut:** ~530 MB

### Aşama 3: Yüz İyileştirme (GFPGAN + seamlessClone)

Bu aşama inswapper_128'in 128x128 pikselleşme sorununu çözmek için tasarlandı:

```python
def enhance_swapped_face(result_img, target_face):
    # 1. Yüz bölgesini %50 padding ile kırp (bağlam için)
    # 2. GFPGAN ile 2x upscale + restore et
    # 3. Orijinal boyuta geri küçült
    # 4. Eliptik mask oluştur
    # 5. cv2.seamlessClone ile doğal geçişle yapıştır
```

**Neden sadece GFPGAN yetmez:**
- GFPGAN tüm görüntüye uygulandığında yüz dışı alanları da değiştirir
- `only_center_face=True` ile sadece merkezdeki yüze odaklanır ama kenar geçişleri sert olur

**Neden seamlessClone:**
- `cv2.NORMAL_CLONE` iyileştirilmiş yüzü orijinal görüntüye renk ve aydınlatma uyumluyla yapıştırır
- Eliptik mask ile yüz kenarlarında yumuşak geçiş sağlanır
- Sonuç: pikselleşme gider, kenarlar doğal görünür

**Pipeline sırası kritik:**
1. Swap sonrası yüzleri **yeniden algıla** (pozisyon değişmiş olabilir)
2. Her yüz için ayrı ayrı iyileştir
3. Her iyileştirmeyi ayrı seamlessClone ile yapıştır

**Model boyut:** ~332 MB

### Denenen ve Elenmiş Yaklaşımlar

| Yaklaşım | Sorun | Sonuç |
|-----------|-------|-------|
| **Sadece inswapper_128** | 128x128 çözünürlük, pikselleşme | Elendi |
| **inswapper_128 + GFPGAN (tüm görüntü)** | Arka plan da değişiyor, gözle görülür fark yok | Elendi |
| **SimSwap 512 (simswap_512_unofficial.onnx)** | Renk/ton uyumsuzluğu, kötü kalite | Elendi |
| **inswapper-512-live** | Kapalı kaynak CoreML, şifreli model, Python'dan kullanılamaz | Elendi |
| **inswapper_128 + GFPGAN (bölgesel) + seamlessClone** | Doğal sonuç, iyi kalite | **Seçildi** |

## Frontend Mimarisi

Tek HTML dosyası (`templates/index.html`), framework kullanmadan vanilla JS ile state yönetimi.

### Tasarım Sistemi

- **Tailwind CSS** (CDN) + özel dark tema renk paleti
- **Space Grotesk** (başlıklar) + **Inter** (gövde) font çifti
- **Material Symbols** ikon seti
- **Glass morphism** efektleri (backdrop-blur, ghost-border)
- Neon cyan (#00f2ff) vurgu rengi, koyu yüzey (#0e0e0e) arka plan

### 4 Adımlı Wizard Akışı

```
State: currentStep (1-4), selectedProfession, selectedGender, selectedTemplate
```

**Adım 1 — Meslek Seçimi:**
- `GET /api/categories` → kategorileri çeker
- Her kategori bir kart, ilk kategori büyük feature card
- Kategoriler dosya sisteminden dinamik okunur (hardcoded değil)

**Adım 2 — Cinsiyet Seçimi:**
- Seçilen mesleğin dosyalarından `kadın` ve `erkek` içerenleri filtreler
- Her cinsiyetten ilk görseli temsili kart olarak gösterir

**Adım 3 — Şablon Seçimi:**
- Meslek + cinsiyet kombinasyonuna göre filtrelenmiş tüm görseller grid'de gösterilir
- Tıklanan şablon seçilir ve otomatik olarak Adım 4'e geçilir

**Adım 4 — Otomatik Çekim + Sonuç:**
- Kullanıcı hiçbir düğmeye basmaz
- Kamera açılır → 5 saniye geri sayım (büyük overlay rakamlar) → otomatik çekim
- Canvas'a çizilir (mirror flip: `ctx.scale(-1, 1)`) → base64 JPEG
- `POST /api/capture` → işlem ekranı (spinner) → sonuç gösterilir
- QR kodu client-side üretilir (qrcodejs kütüphanesi)
- Hata durumunda kamera otomatik yeniden başlar

### Dosya Adı Tabanlı Cinsiyet Filtreleme

Fotoğraf dosya adlarında cinsiyet bilgisi encode edilmiştir:
```
00000-1752474974-kadın.png  → Kız şablonu
00000-2527343689-erkek.png  → Erkek şablonu
```

Frontend'de filtreleme:
```javascript
const imgs = categories[selectedProfession].images.filter(f => f.includes(selectedGender));
```

Bu yaklaşım veritabanı veya metadata dosyası gerektirmez — dosya adı kendisi metadata'dır.

## Backend API

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/` | GET | Web arayüzü (index.html) |
| `/api/status` | GET | Sistem durumu (hangi modeller aktif) |
| `/api/categories` | GET | Meslek kategorileri ve fotoğraf listesi |
| `/api/image/{category}/{filename}` | GET | Şablon fotoğrafı serve et |
| `/api/capture` | POST | Face swap işlemi yap |
| `/api/results` | GET | Son 20 sonucu listele |
| `/api/result/{filename}` | GET | Sonuç fotoğrafını serve et |

### `/api/capture` Request/Response

```json
// Request
{
  "category": "Doktor",
  "image": "00000-2527343689-erkek.png",
  "capturedImage": "data:image/jpeg;base64,..."
}

// Response
{
  "success": true,
  "result": "data:image/jpeg;base64,...",
  "filename": "result_20260410_105005.jpg",
  "download_url": "http://tmpfiles.org/dl/12345/result_20260410_105005.jpg",
  "mode": "inswapper_128+gfpgan+seamless"
}
```

### QR Kod ve Dosya Paylaşımı

Sonuç fotoğrafı `tmpfiles.org` API'sine yüklenir:
- Ücretsiz, kayıt gerektirmez
- 1 saat sonra otomatik silinir
- Doğrudan indirme linki (`/dl/` prefix'i ile)
- QR kodu bu linke yönlenir
- tmpfiles.org erişilemezse fallback olarak lokal `/api/result/` linki kullanılır

## Güvenlik

- **Path traversal koruması:** Tüm dosya yolları `Path(filename).name` ile sanitize edilir
- **Upload limiti:** 50 MB (`MAX_CONTENT_LENGTH`)
- **Lokal çalışma:** `127.0.0.1` bind — sadece localhost'tan erişilebilir
- **Geçici dosyalar:** tmpfiles.org'daki dosyalar 1 saat sonra silinir

## Dizin Yapısı

```
faceswap-app/
├── app.py                  # Flask backend + AI pipeline
├── templates/
│   └── index.html          # Frontend (Tailwind + vanilla JS)
├── uploads/
│   └── Char/               # Şablon fotoğrafları
│       ├── Astronot/       # 31 fotoğraf (11 kız, 18 erkek)
│       ├── Aşçı/           # 15 fotoğraf (7 kız, 8 erkek)
│       ├── Bilim İnsanı/   # 19 fotoğraf (12 kız, 7 erkek)
│       ├── Doktor/         # 22 fotoğraf (13 kız, 9 erkek)
│       ├── İtfaiyeci/      # 14 fotoğraf (7 kız, 7 erkek)
│       ├── Müzisyen/       # 13 fotoğraf (7 kız, 6 erkek)
│       ├── Öğretmen/       # 12 fotoğraf (6 kız, 6 erkek)
│       ├── Pilot/          # 12 fotoğraf (6 kız, 6 erkek)
│       ├── Polis/          # 21 fotoğraf (10 kız, 11 erkek)
│       └── Sporcu/         # 14 fotoğraf (7 kız, 7 erkek)
├── models/                 # AI modelleri (git'e dahil değil)
│   ├── inswapper_128.onnx  # ~530 MB
│   └── GFPGANv1.4.pth     # ~332 MB
├── results/                # Üretilen sonuçlar (git'e dahil değil)
├── requirements.txt        # Python bağımlılıkları
├── download_model.py       # Tüm modelleri indir
├── start.sh                # macOS/Linux başlatma scripti
├── start.bat               # Windows başlatma scripti
└── .gitignore
```

## Kurulum ve Çalıştırma

### Gereksinimler

- Python 3.8+
- Webcam
- Internet bağlantısı (ilk kurulumda model indirme için)

### Hızlı Başlangıç (macOS / Linux)

```bash
git clone https://github.com/gurcay/faceswap-app.git
cd faceswap-app
./start.sh
```

### Hızlı Başlangıç (Windows)

```cmd
git clone https://github.com/gurcay/faceswap-app.git
cd faceswap-app
start.bat
```

Veya `start.bat` dosyasına çift tıklayarak da başlatabilirsiniz.

> **Windows notu:** Python 3.8+ sisteme kurulu olmalı ve PATH'te bulunmalıdır. [python.org](https://www.python.org/downloads/) adresinden indirirken "Add Python to PATH" seçeneğini işaretleyin.

Her iki script de (`start.sh` / `start.bat`) aynı işlemi yapar:
1. Python virtual environment oluşturur
2. Bağımlılıkları kurar (`requirements.txt`)
3. AI modellerini indirir (~900 MB, ilk seferde)
4. `basicsr`/`torchvision` uyumluluk patch'ini uygular
5. Uygulamayı `http://localhost:5001` adresinde başlatır

### Manuel Kurulum

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python download_model.py
python app.py
```

## Yeni Meslek Kategorisi Ekleme

1. `uploads/Char/` altında yeni bir klasör oluştur (ör: `Mühendis`)
2. İçine PNG/JPG fotoğraflar koy
3. Dosya adlarında cinsiyet belirt: `001-erkek.png`, `002-kadın.png`
4. Uygulamayı yeniden başlat — yeni kategori otomatik görünür

Kod tarafında hiçbir değişiklik gerekmez. Kategoriler dosya sisteminden dinamik okunur.

## Bilinen Kısıtlamalar

- **inswapper_128 çözünürlüğü:** Model 128x128'de çalışır. GFPGAN ile iyileştirilse de yüksek çözünürlük swap modelleri (inswapper_512_live gibi) ile kıyaslanamaz. Bu modeller şu an kapalı kaynak.
- **CPU performansı:** GPU olmadan face swap ~3-5 saniye, GFPGAN iyileştirme ~2-3 saniye sürer. Apple Silicon Mac'lerde CoreML provider ile daha hızlı.
- **Tek yüz swap:** Kaynak görüntüden en büyük yüz seçilir. Grup fotoğrafı desteği yok.
- **tmpfiles.org bağımlılığı:** QR indirme linki için internet gerekir. Çevrimdışı ortamda lokal fallback linki kullanılır (aynı ağda olunmalı).

## Lisans

Bu proje eğlence ve eğitim amaçlıdır. Kullanılan AI modelleri kendi lisanslarına tabidir. Başkasının izni olmadan yüz swap yapılmamalıdır.
