import os
import cv2
import numpy as np
import base64
import requests as http_requests
from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime
from pathlib import Path
from insightface.app import FaceAnalysis
import insightface
from gfpgan import GFPGANer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Dizin yapısı
UPLOAD_DIR = Path('uploads')
CATEGORIES_DIR = UPLOAD_DIR / 'Char'
TEMP_DIR = Path('temp')
RESULTS_DIR = Path('results')
MODEL_DIR = Path('models')

for dir_path in [UPLOAD_DIR, CATEGORIES_DIR, TEMP_DIR, RESULTS_DIR, MODEL_DIR]:
    dir_path.mkdir(exist_ok=True)

# ───────────────────────────────────────────────────────
# 1) InsightFace yüz analiz modeli
# ───────────────────────────────────────────────────────
face_analyzer = None
try:
    face_analyzer = FaceAnalysis(
        name='buffalo_l',
        providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'],
        allowed_modules=['detection', 'recognition']
    )
    face_analyzer.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))
    print("✓ InsightFace yüz analiz modeli yüklendi")
except Exception as e:
    print(f"⚠ InsightFace yükleme hatası: {e}")

# ───────────────────────────────────────────────────────
# 2) inswapper_128 face swap modeli
# ───────────────────────────────────────────────────────
face_swapper = None
INSWAPPER_PATH = MODEL_DIR / 'inswapper_128.onnx'

if INSWAPPER_PATH.exists():
    try:
        face_swapper = insightface.model_zoo.get_model(
            str(INSWAPPER_PATH), download=False, download_zip=False
        )
        print("✓ inswapper_128 modeli yüklendi")
    except Exception as e:
        print(f"⚠ inswapper_128 yükleme hatası: {e}")
else:
    print(f"⚠ inswapper_128.onnx bulunamadı ({INSWAPPER_PATH})")

# ───────────────────────────────────────────────────────
# 3) GFPGAN yüz iyileştirme (face restoration)
# ───────────────────────────────────────────────────────
face_restorer = None
GFPGAN_MODEL_PATH = MODEL_DIR / 'GFPGANv1.4.pth'

try:
    face_restorer = GFPGANer(
        model_path=str(GFPGAN_MODEL_PATH) if GFPGAN_MODEL_PATH.exists() else 'GFPGANv1.4.pth',
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None
    )
    print("✓ GFPGAN yüz iyileştirme modeli yüklendi (upscale=2)")
except Exception as e:
    print(f"⚠ GFPGAN yükleme hatası: {e}")


# ───────────────────────────────────────────────────────
# Yüz iyileştirme pipeline
# ───────────────────────────────────────────────────────

def enhance_swapped_face(result_img, target_face):
    """
    Swap edilmiş yüzü GFPGAN ile iyileştir ve seamlessClone ile geri yapıştır.
    Bu yöntem tüm görüntüyü değil, sadece yüz bölgesini işler.
    """
    if face_restorer is None:
        return result_img

    h, w = result_img.shape[:2]

    # Yüz bounding box'ını genişlet (çevre bağlamı için)
    bbox = target_face.bbox.astype(int)
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    pad = int(max(face_w, face_h) * 0.5)

    x1 = max(0, bbox[0] - pad)
    y1 = max(0, bbox[1] - pad)
    x2 = min(w, bbox[2] + pad)
    y2 = min(h, bbox[3] + pad)

    # Yüz bölgesini kırp
    face_region = result_img[y1:y2, x1:x2].copy()

    try:
        # GFPGAN ile iyileştir (upscale=2 -> 2x çözünürlük)
        _, _, restored = face_restorer.enhance(
            face_region, has_aligned=False, only_center_face=True, paste_back=True
        )
        if restored is None:
            return result_img

        # Orijinal boyuta geri küçült
        restored_resized = cv2.resize(restored, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)

        # seamlessClone ile doğal geçiş
        # Yüz merkezini hesapla
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Mask: yüz bölgesini kapsayan elips
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cx_local = (x2 - x1) // 2
        cy_local = (y2 - y1) // 2
        axes_x = int(face_w * 0.55)
        axes_y = int(face_h * 0.65)
        cv2.ellipse(mask, (cx_local, cy_local), (axes_x, axes_y), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 15)

        # seamlessClone
        output = cv2.seamlessClone(
            restored_resized, result_img, mask, (center_x, center_y), cv2.NORMAL_CLONE
        )
        return output

    except Exception as e:
        print(f"Yüz iyileştirme hatası: {e}")
        import traceback
        traceback.print_exc()
        return result_img


# ───────────────────────────────────────────────────────
# Kategori yönetimi
# ───────────────────────────────────────────────────────

def get_categories():
    categories = {}
    if CATEGORIES_DIR.exists():
        for cat_dir in CATEGORIES_DIR.iterdir():
            if cat_dir.is_dir():
                images = (
                    list(cat_dir.glob('*.jpg')) +
                    list(cat_dir.glob('*.jpeg')) +
                    list(cat_dir.glob('*.png'))
                )
                if images:
                    categories[cat_dir.name] = {
                        'label': cat_dir.name,
                        'images': [img.name for img in sorted(images)]
                    }
    return categories


# ───────────────────────────────────────────────────────
# Ana face swap fonksiyonu
# ───────────────────────────────────────────────────────

def do_face_swap(source_img, target_img):
    """inswapper_128 ile swap, GFPGAN ile iyileştirme, seamlessClone ile yapıştırma"""
    if face_analyzer is None:
        raise RuntimeError("Yüz analiz modeli yüklenemedi")
    if face_swapper is None:
        raise RuntimeError("Face swap modeli yüklenemedi")

    source_faces = face_analyzer.get(source_img)
    target_faces = face_analyzer.get(target_img)

    if not source_faces:
        raise ValueError("Kaynak görüntüde yüz bulunamadı. Lütfen net ve aydınlık bir fotoğraf çekin.")
    if not target_faces:
        raise ValueError("Hedef fotoğrafta yüz bulunamadı.")

    # En büyük yüzü al
    source_face = max(source_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # 1) inswapper_128 ile face swap
    result = target_img.copy()
    for target_face in target_faces:
        result = face_swapper.get(result, target_face, source_face, paste_back=True)

    # 2) Her yüz için GFPGAN iyileştirme + seamlessClone
    if face_restorer is not None:
        # Swap sonrası yüzleri yeniden algıla (pozisyon değişmiş olabilir)
        result_faces = face_analyzer.get(result)
        if result_faces:
            for face in result_faces:
                result = enhance_swapped_face(result, face)

    mode = "inswapper_128"
    if face_restorer is not None:
        mode += "+gfpgan+seamless"

    return result, mode


# ───────────────────────────────────────────────────────
# Flask routes
# ───────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'face_analyzer': face_analyzer is not None,
        'face_swapper': face_swapper is not None,
        'face_restorer': face_restorer is not None,
        'swap_mode': 'inswapper_128 + GFPGAN + seamlessClone' if (face_swapper and face_restorer) else
                     'inswapper_128' if face_swapper else 'Model yok'
    })


@app.route('/api/categories', methods=['GET'])
def api_categories():
    return jsonify(get_categories())


@app.route('/api/image/<category>/<filename>', methods=['GET'])
def api_image(category, filename):
    filename = Path(filename).name
    category = Path(category).name
    image_path = CATEGORIES_DIR / category / filename
    if image_path.exists():
        suffix = image_path.suffix.lower()
        mime = 'image/jpeg' if suffix in ('.jpg', '.jpeg') else 'image/png'
        return send_file(image_path, mimetype=mime)
    return jsonify({'error': 'Fotoğraf bulunamadı'}), 404


@app.route('/api/capture', methods=['POST'])
def api_capture():
    try:
        data = request.json
        category = data.get('category', '')
        filename = data.get('image', '')
        image_data = data.get('capturedImage', '')

        if not image_data.startswith('data:image'):
            return jsonify({'error': 'Geçersiz görüntü formatı'}), 400

        _, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        source_array = np.frombuffer(image_bytes, dtype=np.uint8)
        source_img = cv2.imdecode(source_array, cv2.IMREAD_COLOR)

        if source_img is None:
            return jsonify({'error': 'Görüntü okunamadı'}), 400

        filename = Path(filename).name
        category = Path(category).name
        target_path = CATEGORIES_DIR / category / filename
        if not target_path.exists():
            return jsonify({'error': 'Hedef fotoğraf bulunamadı'}), 404

        target_img = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if target_img is None:
            return jsonify({'error': 'Hedef fotoğraf okunamadı'}), 400

        result_img, mode = do_face_swap(source_img, target_img)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_filename = f'result_{timestamp}.jpg'
        result_path = RESULTS_DIR / result_filename
        cv2.imwrite(str(result_path), result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result_b64 = base64.b64encode(buffer).decode()

        # tmpfiles.org'a yükle (1 saat sonra otomatik silinir)
        download_url = None
        try:
            with open(result_path, 'rb') as f:
                resp = http_requests.post(
                    'https://tmpfiles.org/api/v1/upload',
                    files={'file': (result_filename, f, 'image/jpeg')},
                    timeout=20
                )
            if resp.status_code == 200:
                resp_data = resp.json()
                if resp_data.get('status') == 'success':
                    # URL'yi doğrudan indirme linkine çevir
                    url = resp_data['data']['url']
                    download_url = url.replace('tmpfiles.org/', 'tmpfiles.org/dl/')
                    print(f"✓ tmpfiles.org yüklendi: {download_url}")
        except Exception as e:
            print(f"⚠ tmpfiles.org yükleme hatası: {e}")

        return jsonify({
            'success': True,
            'result': f'data:image/jpeg;base64,{result_b64}',
            'filename': result_filename,
            'download_url': download_url,
            'mode': mode
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Hata: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'İşlem hatası: {str(e)}'}), 500


@app.route('/api/results', methods=['GET'])
def api_results():
    results = []
    for result_file in sorted(RESULTS_DIR.glob('*.jpg'), reverse=True)[:20]:
        results.append({
            'filename': result_file.name,
            'url': f'/api/result/{result_file.name}'
        })
    return jsonify(results)


@app.route('/api/result/<filename>', methods=['GET'])
def api_result(filename):
    filename = Path(filename).name
    result_path = RESULTS_DIR / filename
    if result_path.exists():
        return send_file(result_path, mimetype='image/jpeg')
    return jsonify({'error': 'Sonuç bulunamadı'}), 404


if __name__ == '__main__':
    print("\n🚀 FaceSwap uygulaması başlatılıyor...")
    print(f"   Yüz analizi:     {'✓ Aktif' if face_analyzer else '✗ Hata'}")
    print(f"   Face swap:       {'✓ inswapper_128' if face_swapper else '✗ Model yok'}")
    print(f"   Yüz iyileştirme: {'✓ GFPGAN + seamlessClone' if face_restorer else '✗ GFPGAN yok'}")
    print("\n📍 http://localhost:5001\n")
    app.run(debug=False, host='127.0.0.1', port=5001, threaded=True)
