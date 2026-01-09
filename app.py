# app.py

import os
# --- SİSTEM AYARLARI ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time
import numpy as np

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="ZeytinPro AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS TASARIMI (Görsel İyileştirmeler) ---
st.markdown("""
    <style>
    /* Font ve Genel Ayarlar */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Roboto', sans-serif; }
    .stApp { background-color: #f4f6f9; }

    /* Üst Başlık */
    .header-container {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        text-align: center;
        border-bottom: 4px solid #2E7D32;
    }
    .header-title { color: #2E7D32; font-size: 2rem; font-weight: 800; margin: 0; }
    .header-subtitle { color: #666; font-size: 1rem; margin-top: 5px; }

    /* Sonuç Kartı */
    .result-card {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        border-top: 8px solid #2E7D32;
        margin-top: 10px;
    }
    .result-label { font-size: 0.9rem; color: #888; letter-spacing: 2px; text-transform: uppercase; }
    .result-value { font-size: 2.5rem; font-weight: 900; color: #1b5e20; line-height: 1.2; margin-top: 10px; }

    /* Bilgi Kutuları */
    .info-box { padding: 20px; border-radius: 10px; margin-top: 15px; font-size: 1rem; line-height: 1.5; }
    .box-success { background-color: #e8f5e9; border-left: 5px solid #2E7D32; color: #1b5e20; }
    .box-warning { background-color: #fff3e0; border-left: 5px solid #ef6c00; color: #e65100; }
    
    /* Resim Kartları */
    .img-card { background: white; padding: 10px; border-radius: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center; margin-bottom: 10px; }
    .img-caption { font-weight: bold; color: #555; margin-bottom: 8px; display: block; }
    
    /* Yan Menü Resimleri */
    [data-testid="stSidebar"] img {
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-bottom: 15px;
    }
    
    /* Buton */
    .stButton>button {
        width: 100%; height: 55px; border-radius: 10px;
        background: linear-gradient(90deg, #2E7D32 0%, #43a047 100%);
        color: white; font-weight: bold; font-size: 1.2rem; border: none;
        box-shadow: 0 4px 10px rgba(46, 125, 50, 0.3);
    }
    .stButton>button:hover { box-shadow: 0 6px 15px rgba(46, 125, 50, 0.4); }
    </style>
""", unsafe_allow_html=True)

# --- 3. VERİTABANI ---
DISEASE_INFO = {
    "olive_peacock_spot": {
        "name": "HALKALI LEKE",
        "desc": "Mantar kaynaklı bir enfeksiyondur. Yapraklarda halkalar oluşturur.",
        "treatment": "Hasat sonrası ve ilkbaharda **Bordo Bulamacı** uygulanmalıdır."
    },
    "Aculus_olearius": {
        "name": "ZEYTİN PAS AKARI",
        "desc": "Mikroskobik zararlılardır. Yaprakta pas rengi ve şekil bozukluğu yapar.",
        "treatment": "İlkbaharda **kükürtlü bileşikler** veya akarisit kullanılmalıdır."
    },
    "healthy": {
        "name": "SAĞLIKLI / TEMİZ",
        "desc": "Herhangi bir hastalık belirtisine rastlanmamıştır.",
        "treatment": "Rutin bakım (sulama, gübreleme) işlemlerine devam ediniz."
    },
    "Unknown": {
        "name": "TANIMLANAMADI",
        "desc": "Görüntü net değil.",
        "treatment": "Lütfen daha net bir fotoğraf çekiniz."
    }
}

MODEL_DATA = {
    "Model 1": {"path": "best1.pt", "weight": 0.993},
    "Model 2": {"path": "best2.pt", "weight": 0.950},
    "Model 3": {"path": "best3.pt", "weight": 0.975}
}
LEAF_MODEL_PATH = "best4.pt"
CLASS_NAMES = ['Aculus_olearius', 'healthy', 'olive_peacock_spot', 'Unknown']

# --- 4. MODEL YÜKLEME ---
@st.cache_resource
def load_models():
    MODELS = {}
    try:
        for name, data in MODEL_DATA.items():
            if os.path.exists(data['path']):
                MODELS[name] = YOLO(data['path'])
        if os.path.exists(LEAF_MODEL_PATH):
            leaf_model = YOLO(LEAF_MODEL_PATH)
            return MODELS, leaf_model
    except: return {}, None
    return {}, None

MODELS, LEAF_MODEL = load_models()

# --- 5. ANALİZ MANTIĞI ---
def run_analysis(img):
    # max_det=1 -> SADECE 1 KUTU
    res = LEAF_MODEL(img, verbose=False, conf=0.85, max_det=1)[0]
    box_img = None
    is_leaf = False
    
    if res.boxes and len(res.boxes) > 0:
        is_leaf = True
        # Labels=True, Conf=True -> Resim üzerinde yazı yazar
        plot_arr = res.plot(labels=True, conf=True, line_width=3, font_size=1.0) 
        box_img = Image.fromarray(plot_arr[..., ::-1]) 
    
    if not is_leaf: return None

    votes = {}
    for name, model in MODELS.items():
        r = model(img, verbose=False)[0]
        if r.probs:
            conf = float(r.probs.top1conf)
            lbl = CLASS_NAMES[r.probs.top1] if conf > 0.55 else "Unknown"
            w = MODEL_DATA[name]["weight"]
            if lbl == "Unknown": w *= 0.2
            votes[lbl] = votes.get(lbl, 0) + w
            
    best_class = max(votes, key=votes.get)
    return {"class": best_class, "box_img": box_img}

# --- 6. YAN MENÜ (SIDEBAR) - YEREL RESİMLER ---
with st.sidebar:
    # 1. ÜST LOGO (olive.jpg)
    if os.path.exists("olive.jpg"):
        st.image("olive.jpg", use_container_width=True)
    else:
        st.warning("⚠️ 'olive.jpg' bulunamadı.")
    
    st.title("ZeytinPro AI")
    st.markdown("Dashboard Kontrolü")
    
    # KONTROLLER
    uploaded_file = st.file_uploader("Görüntü Yükle", type=['jpg', 'png', 'jpeg'])
    analyze_btn = st.button("ANALİZİ BAŞLAT")
    
    st.markdown("---")
    
    # 2. ALT GÖRSEL (farmer.jpg)
    if os.path.exists("farmer.jpg"):
        st.image("farmer.jpg", caption="Çiftçi Dostu Teknoloji", use_container_width=True)
    
    st.markdown("---")
    st.info("Geliştirici: **Mümtaz Şahin Delibaş**")

# --- 7. ANA EKRAN ---
st.markdown("""
    <div class="header-container">
        <div class="header-title">🌿 YAPAY ZEKA DESTEKLİ ZEYTİN TANI SİSTEMİ</div>
        <div class="header-subtitle">Görüntü İşleme ve Derin Öğrenme Teknolojileri</div>
    </div>
""", unsafe_allow_html=True)

if uploaded_file:
    if 'last_file' not in st.session_state or st.session_state['last_file'] != uploaded_file.name:
        st.session_state['processed'] = False
        st.session_state['last_file'] = uploaded_file.name
        st.session_state['result'] = None

    image_pil = Image.open(uploaded_file).convert("RGB")
    
    col_left, col_right = st.columns([6, 5], gap="large")
    
    # SOL KOLON (GÖRSELLER)
    with col_left:
        if st.session_state['processed'] and st.session_state['result'] and st.session_state['result']['box_img']:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="img-card"><span class="img-caption">1. Orijinal</span></div>', unsafe_allow_html=True)
                st.image(image_pil, use_container_width=True)
            with c2:
                st.markdown('<div class="img-card"><span class="img-caption">2. AI Tespiti</span></div>', unsafe_allow_html=True)
                st.image(st.session_state['result']['box_img'], use_container_width=True)
        else:
            st.markdown('<div class="img-card"><span class="img-caption">Yüklenen Numune</span></div>', unsafe_allow_html=True)
            st.image(image_pil, use_container_width=True)

    # SAĞ KOLON (RAPOR)
    with col_right:
        if analyze_btn:
            with st.spinner("Modeller çalışıyor..."):
                time.sleep(0.5)
                res = run_analysis(image_pil)
                st.session_state['result'] = res
                st.session_state['processed'] = True
                st.rerun()
        
        if st.session_state['processed']:
            res = st.session_state['result']
            
            if not res:
                st.error("⚠️ Yaprak tespit edilemedi.")
                st.warning("Lütfen daha net bir fotoğraf yükleyiniz.")
            else:
                info = DISEASE_INFO.get(res['class'], DISEASE_INFO['Unknown'])
                
                theme_color = "#2E7D32" if res['class'] == 'healthy' else ("#757575" if res['class'] == 'Unknown' else "#d32f2f")

                # SONUÇ KARTI (Yüzde YOK)
                st.markdown(f"""
                <div class="result-card" style="border-top-color: {theme_color};">
                    <div class="result-label">TESPİT EDİLEN DURUM</div>
                    <div class="result-value" style="color: {theme_color};">{info['name']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # ÖNERİ
                st.markdown(f"""
                <div class="info-box box-success">
                    <strong>💡 Zirai Tavsiye:</strong><br>
                    {info['treatment']}
                    <br><br>
                    <small style="color:#555"><em>Teşhis Bilgisi: {info['desc']}</em></small>
                </div>
                """, unsafe_allow_html=True)
                
                # UYARI
                st.markdown("""
                <div class="info-box box-warning">
                    ⚠️ <strong>Yasal Uyarı:</strong> Bu rapor ön bilgilendirmedir. Kesin teşhis için <strong>Tarım Müdürlüğü</strong>'ne başvurunuz.
                </div>
                """, unsafe_allow_html=True)

        else:
            st.info("👈 Analizi başlatmak için sol taraftaki butona basınız.")

else:
    st.info("Başlamak için lütfen sol menüden bir yaprak fotoğrafı yükleyiniz.")
