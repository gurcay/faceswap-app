# FaceSwap - Lokal Yüz Değiştirme Uygulaması

Eğlence amaçlı, lokal olarak çalışan bir web tabanlı yüz değiştirme (faceswap) uygulaması.

## 🎯 Özellikler

✅ **Web Arayüzü** - Modern, kullanıcı dostu interface
✅ **Gerçek Zamanlı Kamera** - Tarayıcıdan doğrudan kamera erişimi
✅ **Otomatik Fotoğraf Çekme** - 5 saniye sonra otomatikmente çekim
✅ **Meslek Kategorileri** - Doktor, Pilot, Astronot vb. kategoriler
✅ **GPU Optimizasyonu** - Apple M1 üzerinde hızlı işlem
✅ **Sonuç İndirme** - Oluşturulan görüntüleri indir
✅ **Geçmiş Sonuçlar** - En son sonuçları görüntüle

## 💻 Sistem Gereksinimleri

- macOS 11+
- Python 3.8+
- 8GB RAM (M1 Mac'ler için optimize)

## 📦 Kurulum

### 1. Virtual Environment Oluştur
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

**Not**: İlk çalıştırmada InsightFace modeli (~500MB) otomatikmente indirilecek.

### 3. Kategori Fotoğraflarını Hazırla

Aşağıdaki struktur içinde fotoğrafları düzenleyin:
```
uploads/categories/
├── doctor/
│   ├── doc1.jpg
│   ├── doc2.jpg
│   └── ...
├── pilot/
│   ├── pilot1.jpg
│   ├── pilot2.jpg
│   └── ...
├── astronaut/
├── superhero/
├── musician/
├── ...
```

**Her kategori en az 1 fotoğraf içermelidir.**

### 4. Uygulamayı Başlat
```bash
python app.py
```

Tarayıcıda açın: **http://localhost:5000**

## 🎬 Kullanım Senaryosu

1. **Fotoğraf Seç**: Soldaki panelden kategori ve fotoğraf seç
2. **Kamerayla Çek**: Sağ paneldeki "Başla" düğmesine tıkla
3. **Otomatik İşlem**: 5 saniye sonra fotoğraf otomatikmente çekilir
4. **Sonuç Göster**: Yüz değiştirilmiş fotoğraf gösterilir
5. **İndir**: Sonuç fotoğrafını indir veya yeniden dene

## 📸 Örnek Kategori Oluşturma

### Hızlı Test Kategorileri Oluştur
```bash
python setup_test_categories.py
```

Bu script test kategorileri ve örnek fotoğrafları oluşturacaktır.

## 🔧 Gelişmiş Yapılandırma

### M1 Mac Optimizasyonu
App otomatikmente CoreML ve CPU execution provider kullanır. Performans için MPSGraph provider de eklenebilir.

### Kamera Ayarları (app.py içinde)
```python
video: {
    width: { ideal: 1280 },
    height: { ideal: 720 }
}
```

### Blending Kalitesi
templates/index.html içinde blur level ayarlanabilir:
```javascript
cv2.GaussianBlur(mask, (51, 51), 30)  // (çekirdek boyutu, std sapma)
```

## 📁 Proje Yapısı

```
faceswap/
├── app.py                    # Flask backend
├── requirements.txt          # Python bağımlılıkları
├── templates/
│   └── index.html           # Web arayüzü
├── uploads/
│   └── categories/          # Kategori fotoğrafları
├── temp/                    # Geçici dosyalar
├── results/                 # İşlenmiş sonuçlar
└── README.md
```

## ⚠️ İlk Çalıştırma Notları

1. **İlk Model İndirmesi**: İlk çalıştırmada 2-3 dakika sürebilir
2. **Kamera İzni**: İlk kullanımda tarayıcı kamera izni isteyecek
3. **Port Kullanımı**: 5000 numaralı port kullanılır

## 🐛 Sorun Giderme

### Kamera Çalışmıyor
- Chrome/Safari kullanın (Edge sorun yaşayabilir)
- HTTPS kullanılmıyorsa localhost:5000'de çalıştığından emin olun
- Tarayıcı ayarlarında kamera izni kontrol edin

### Yüz Tanınmıyor
- Fotoğraf açık ve net olması gereklidir
- En az 640x480 çözünürlükte fotoğraf kullanın
- Yüz boyutu yeterince büyük olmalıdır

### Yavaş İşlem
- M1 Mac'lerde ilk sefer yavaş olabilir (model optimize ediliyor)
- Sonraki işlemler daha hızlı olacaktır
- Tarayıcı JavaScript console'da (F12) hataları kontrol edin

## 📝 API Endpoints

- `GET /` - Web arayüzü
- `GET /api/categories` - Kategorileri getir
- `GET /api/image/<category>/<filename>` - Kategori fotoğrafı
- `POST /api/capture` - Yüz değiştirme işlemi
- `GET /api/results` - Son sonuçları getir
- `GET /api/result/<filename>` - Sonuç fotoğrafı

## 🎨 Özelleştirme

### Yeni Kategori Ekleme
`uploads/categories/` altına klasör oluşturun ve fotoğraf ekleyin.

### Arayüz Renkleri (HTML içinde)
```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

## 🚀 Performans İpuçları (M1 Mac)

- İlk kez chạy slow olabilir (model optimization)
- Safari Chrome'dan daha hızlı olabilir
- 1280x720 ideal çözünürlüktür
- 5 saniye timer optimal değerdir

## ⚖️ Yasal Uyarı

Bu yazılım **yalnızca eğlence amaçlı** kullanılmalıdır. Başkasının rızası olmadan yüzünü kullanan içerik oluşturmayın.

## 📞 Destek

Sorunlar için `app.py`'daki logging çıktısını kontrol edin ve hata mesajlarını not edin.

---

**Geliştirici**: Claude Code AI
**Sürüm**: 1.0
**Güncelleme**: 2026-04-08
