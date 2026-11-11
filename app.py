import os
import io
import glob
import joblib
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_curve, auc)

st.set_page_config(
    page_title="DETAY | Deri Tanı ve Ön Triyaj Sistemi",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700;800;900&display=swap');
html, body, [class*="st-"], .stMarkdown, button, input, label { font-family: 'Nunito', 'Segoe UI', sans-serif !important; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }
.stMarkdown p, .stMarkdown li, label, .stRadio label, .stAlert p { font-size: 1.2rem !important; line-height: 1.6; }
h3 { font-size: 1.9rem !important; font-weight: 900 !important; }
h4 { font-size: 1.45rem !important; font-weight: 800 !important; }
.stTabs [data-baseweb="tab"] p { font-size: 1.25rem !important; font-weight: 800; }
[data-testid="stFileUploader"] section { padding: 1.2rem; border: 3px dashed #7C4DFF; border-radius: 18px; }
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { font-size: 1.05rem !important; }
[data-testid="stImage"] img { border-radius: 18px; max-height: 560px; object-fit: contain; }
[data-testid="stCaptionContainer"] p { font-size: 1.05rem !important; }

.hero { background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 45%, #EC4899 100%); border-radius: 24px;
        padding: 28px 36px; color: #FFFFFF !important; box-shadow: 0 10px 30px rgba(79,70,229,.35); margin-bottom: 8px; }
.hero h1 { color: #FFFFFF !important; font-size: 3rem; font-weight: 900; margin: 0 0 6px 0; line-height: 1.15; }
.hero p  { color: #F3E8FF !important; font-size: 1.25rem !important; margin: 4px 0; }
.chip { display: inline-block; background: rgba(255,255,255,.22); color: #FFFFFF !important; padding: 6px 16px;
        border-radius: 999px; font-weight: 800; font-size: 1.05rem; margin: 6px 8px 0 0; }

.sonuc-kart { border-radius: 22px; padding: 26px 30px; margin: 10px 0 18px 0; color: #FFFFFF !important; box-shadow: 0 8px 24px rgba(0,0,0,.22); }
.sonuc-kart .emoji { font-size: 3.2rem; line-height: 1; }
.sonuc-kart h2 { color: #FFFFFF !important; font-size: 2.3rem; font-weight: 900; margin: 6px 0 12px 0; line-height: 1.2; }
.sonuc-kart .satir { color: #FFFFFF !important; font-size: 1.3rem; font-weight: 700; margin: 6px 0; }
.sonuc-kart .rozet { display: inline-block; background: rgba(255,255,255,.95); color: #1F2937 !important; padding: 6px 16px;
        border-radius: 999px; font-weight: 900; font-size: 1.1rem; margin: 4px 8px 4px 0; }

.bilgi-kutu { background: #EEF2FF; border: 2px solid #C7D2FE; border-radius: 18px; padding: 16px 22px; margin: 10px 0; color: #1E1B4B !important; font-size: 1.2rem; line-height: 1.6; }
.bilgi-kutu b { color: #3730A3 !important; }
.uyari-kutu { background: #FFFBEB; border: 2px solid #FCD34D; border-radius: 18px; padding: 16px 22px; margin: 14px 0; color: #78350F !important; font-size: 1.1rem; line-height: 1.55; }
.cocuk-kutu { background: #ECFDF5; border: 2px solid #6EE7B7; border-radius: 18px; padding: 16px 22px; margin: 10px 0; color: #064E3B !important; font-size: 1.2rem; line-height: 1.6; }

.bar-satir { margin: 14px 0; }
.bar-ad { font-size: 1.3rem; font-weight: 800; color: #111827 !important; margin-bottom: 4px; }
.bar-dis { background: #E5E7EB; border-radius: 999px; height: 34px; overflow: hidden; }
.bar-ic { height: 34px; border-radius: 999px; color: #FFFFFF !important; font-weight: 900; font-size: 1.15rem;
          display: flex; align-items: center; justify-content: flex-end; padding-right: 12px; min-width: 70px; }

.metrik { background: linear-gradient(135deg,#EEF2FF,#FDF2F8); border: 2px solid #E0E7FF; border-radius: 20px; padding: 18px; text-align: center; }
.metrik .deger { font-size: 2.4rem; font-weight: 900; color: #4338CA !important; line-height: 1.1; }
.metrik .etiket { font-size: 1.05rem; font-weight: 700; color: #4B5563 !important; }
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))
ARAMA_DIZINLERI = [
    BASE,
    os.path.join(BASE, "sonuclar"),
    os.path.join(BASE, "sonuclar", "modeller"),
    os.path.join(BASE, "modeller"),
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dosya_bul(desenler):
    for d in ARAMA_DIZINLERI:
        if not os.path.exists(d):
            continue
        for desen in desenler:
            eslesen = sorted(glob.glob(os.path.join(d, desen)))
            if eslesen:
                return eslesen[0]
    return None

YUKSEK_ESIK = 70.0
BELIRSIZ_ESIK = 40.0
MIN_FARK = 10.0

SINIF_BILGI = {
    "Chickenpox": {
        "tr_ad": "Su Çiçeği (Chickenpox)", "emoji": "🫧", "renk": "#F97316", "renk2": "#FB923C",
        "risk": "Orta Düzey Bulaşıcılık Riski",
        "aciklama": "Varicella-zoster virüsünün neden olduğu, kaşıntılı ve sıvı dolu veziküllerle seyreden döküntü.",
        "oneri": "Lezyonlar tamamen kabuklanana kadar toplu alan/okul izolasyonu önerilir. Aile hekimine başvurunuz.",
        "kaynak": "T.C. Sağlık Bakanlığı HSGM Su Çiçeği (Varisella) Saha Rehberi (2023); CDC Pink Book: Varicella (2024).",
        "cocuk": "Su çiçeğinde cildin üzerinde içi sıvı dolu, kaşınan küçük kabarcıklar çıkar."
    },
    "Healthy": {
        "tr_ad": "Sağlıklı Deri (Healthy)", "emoji": "🌿", "renk": "#16A34A", "renk2": "#4ADE80",
        "risk": "Enfeksiyöz Bulgusu Saptanmadı",
        "aciklama": "İncelenen cilt bölgesinde viral döküntü veya patolojik lezyon örüntüsü görülmemiştir.",
        "oneri": "Genel cilt sağlığı için hijyen kurallarına uymaya devam ediniz. Ateş veya yeni döküntü gelişirse hekime başvurunuz.",
        "kaynak": "WHO Syndromic Triage Guidelines for Acute Rash Illness (2022); T.C. Sağlık Bakanlığı Birinci Basamak Triyaj İlkeleri.",
        "cocuk": "Bu fotoğrafta bu hastalıklara özgü bir döküntü görünmüyor. Ellerini yıkamaya devam!"
    },
    "Measles": {
        "tr_ad": "Kızamık (Measles)", "emoji": "🔴", "renk": "#7C3AED", "renk2": "#A78BFA",
        "risk": "Yüksek Bulaşıcılık Riski (Solunum Yolu)",
        "aciklama": "Cilde yayılan makülopapüler kırmızı-pembe lekeler şeklinde viral döküntü örüntüsü.",
        "oneri": "Yüksek solunum bulaşıcılığı nedeniyle maske takılmalı ve ivedi olarak sağlık kuruluşuna başvurulmalıdır.",
        "kaynak": "T.C. Sağlık Bakanlığı HSGM Kızamık Saha Rehberi (2023); WHO Measles Surveillance Standards (2023).",
        "cocuk": "Kızamıkta cildin üzerinde kırmızı-pembe lekeler yayılır ve hastalık öksürük, hapşırıkla kolayca bulaşır."
    },
    "Monkeypox": {
        "tr_ad": "Maymun Çiçeği (Mpox)", "emoji": "🐵", "renk": "#DC2626", "renk2": "#F87171",
        "risk": "Yüksek Klinik Öncelik (Temas İzolasyonu Gerekli)",
        "aciklama": "Derin yerleşimli, göbekli (umbilike) vezikül veya püstül morfolojisinde döküntü.",
        "oneri": "Doğrudan temastan kaçınılmalı, kişisel eşyalar ayrılmalı ve PCR doğrulaması için hastaneye gidilmelidir.",
        "kaynak": "T.C. Sağlık Bakanlığı HSGM Mpox (Maymun Çiçeği) Rehberi (2024); WHO Clinical Management for Monkeypox (2022); CDC (2024).",
        "cocuk": "Maymun çiçeğinde ortası çukur görünen kabarcıklar oluşabilir. Kesin tanıyı sadece doktor koyar."
    },
}


@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@st.cache_resource
def load_backbone(backbone_name: str):
    if backbone_name == "ConvNeXt-Tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Identity()
    elif backbone_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier = nn.Identity()
    elif backbone_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
    else:
        raise ValueError(f"Bilinmeyen backbone: {backbone_name}")
    model = model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_sampiyon_model():
    yol = dosya_bul(["sampiyon_model.joblib", "model_ConvNeXt-Tiny_SVM.joblib", "*.joblib"])
    if yol and os.path.exists(yol):
        return joblib.load(yol)
    return None


def acc_metni(paket):
    for anahtar in ("test_acc", "test_accuracy", "accuracy"):
        if anahtar in paket:
            v = float(paket[anahtar])
            return f"%{v * 100:.2f}" if v <= 1 else f"%{v:.2f}"
    return "%89.30"


@torch.no_grad()
def occlusion_haritasi(backbone, clf, scaler, x, hedef_idx, grid=7):
    _, _, H, W = x.shape
    sh, sw = H // grid, W // grid

    def prob(t):
        f = backbone(t).cpu().numpy()
        return clf.predict_proba(scaler.transform(f))[:, hedef_idx]

    temel = prob(x)[0]
    varyantlar = []
    for i in range(grid):
        for j in range(grid):
            v = x.clone()
            v[:, :, i * sh:(i + 1) * sh, j * sw:(j + 1) * sw] = 0.0
            varyantlar.append(v)
    olas = prob(torch.cat(varyantlar, 0))
    onem = np.clip(temel - olas, 0, None).reshape(grid, grid)
    if onem.max() > 0:
        onem = onem / onem.max()
    return onem


def kart_html(emoji, baslik, satirlar, rozetler, renk1, renk2):
    satir_html = "".join(f'<div class="satir">{s}</div>' for s in satirlar)
    rozet_html = "".join(f'<span class="rozet">{r}</span>' for r in rozetler)
    return f"""<div class="sonuc-kart" style="background:linear-gradient(135deg,{renk1},{renk2});">
        <div class="emoji">{emoji}</div><h2>{baslik}</h2><div>{rozet_html}</div>{satir_html}</div>"""


def metrik_kutusu(deger, etiket):
    return f'<div class="metrik"><div class="deger">{deger}</div><div class="etiket">{etiket}</div></div>'


SINIF_SIRA = ["Chickenpox", "Healthy", "Measles", "Monkeypox"]
KISA_AD = {"Chickenpox": "Su Çiçeği", "Healthy": "Sağlıklı", "Measles": "Kızamık", "Monkeypox": "Maymun Çiçeği"}
RENK = {"Chickenpox": "#F97316", "Healthy": "#16A34A", "Measles": "#7C3AED", "Monkeypox": "#DC2626"}


def kolon_bul(df, anahtarlar):
    for k in df.columns:
        if all(a.lower() in k.lower() for a in anahtarlar):
            return k
    return None


def tahmin_verisi_hazirla(df):
    df = df.copy()
    mk = kolon_bul(df, ["Model"])
    if mk is not None and df[mk].nunique() > 1:
        sec = [m for m in df[mk].unique() if "convnext" in str(m).lower() and "svm" in str(m).lower()]
        df = df[df[mk] == (sec[0] if sec else df[mk].iloc[0])]
    gercek = df[kolon_bul(df, ["Gercek"])].map({c: i for i, c in enumerate(SINIF_SIRA)}).to_numpy()
    tahmin = df[kolon_bul(df, ["Tahmin_Sinif"])].map({c: i for i, c in enumerate(SINIF_SIRA)}).to_numpy()
    olas = np.column_stack([df[f"Olasilik_{c}"].to_numpy(dtype=float) for c in SINIF_SIRA])
    return gercek.astype(int), tahmin.astype(int), olas


def fig_kaydet_goster(fig, dosya_adi, sekil_no, aciklama):
    st.pyplot(fig)
    st.caption(f"**Şekil {sekil_no}.** {aciklama}")
    tampon = io.BytesIO()
    fig.savefig(tampon, format="png", dpi=300, bbox_inches="tight")
    st.download_button("⬇️ Yüksek çözünürlüklü PNG indir (300 dpi)", tampon.getvalue(),
                       file_name=dosya_adi, mime="image/png", key=f"dl_{sekil_no}")
    plt.close(fig)


def stil(ax, baslik, xl=None, yl=None):
    ax.set_title(baslik, fontsize=14, fontweight="bold", pad=10)
    if xl: ax.set_xlabel(xl, fontsize=12)
    if yl: ax.set_ylabel(yl, fontsize=12)
    ax.tick_params(labelsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def cm_grafigi(y, p):
    cm = confusion_matrix(y, p, labels=range(4))
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ad = [KISA_AD[c] for c in SINIF_SIRA]
    ax.set_xticks(range(4)); ax.set_xticklabels(ad, rotation=20)
    ax.set_yticks(range(4)); ax.set_yticklabels(ad)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n(%{norm[i, j]*100:.0f})", ha="center", va="center", fontsize=12,
                    color="white" if norm[i, j] > 0.5 else "black", fontweight="bold")
    stil(ax, "Karışıklık Matrisi", "Tahmin edilen sınıf", "Gerçek sınıf")
    for s in ax.spines.values(): s.set_visible(True)
    fig.tight_layout()
    return fig


def metrik_tablosu(y, p):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(4), zero_division=0)
    t = pd.DataFrame({"Sınıf": [KISA_AD[c] for c in SINIF_SIRA], "Kesinlik": pr, "Duyarlılık": rc,
                      "F1": f1, "Örnek sayısı": sup})
    return t


def metrik_grafigi(t):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(t)); w = 0.26
    for k, (kol, renk) in enumerate([("Kesinlik", "#6366F1"), ("Duyarlılık", "#EC4899"), ("F1", "#14B8A6")]):
        b = ax.bar(x + (k - 1) * w, t[kol], w, label=kol, color=renk, edgecolor="black", linewidth=0.6)
        for r in b:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.2f}",
                    ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(t["Sınıf"]); ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    stil(ax, "Sınıf Bazında Başarım", None, "Skor")
    fig.tight_layout()
    return fig


def roc_grafigi(y, olas):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, c in enumerate(SINIF_SIRA):
        yt = (y == i).astype(int)
        if yt.sum() == 0: continue
        fpr, tpr, _ = roc_curve(yt, olas[:, i])
        ax.plot(fpr, tpr, lw=2.5, color=RENK[c], label=f"{KISA_AD[c]} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    ax.grid(linestyle="--", alpha=0.4)
    stil(ax, "ROC Eğrileri (Bire Karşı Hepsi)", "Yanlış pozitif oranı", "Doğru pozitif oranı")
    fig.tight_layout()
    return fig


def guven_dagilimi_grafigi(dogru, guven):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bins = np.linspace(0, 1, 11)
    ax.hist(guven[dogru], bins=bins, alpha=0.8, color="#16A34A", edgecolor="black", label=f"Doğru tahmin (n={dogru.sum()})")
    if (~dogru).sum() > 0:
        ax.hist(guven[~dogru], bins=bins, alpha=0.85, color="#DC2626", edgecolor="black", label=f"Yanlış tahmin (n={(~dogru).sum()})")
    ax.axvline(0.40, color="#F59E0B", ls="--", lw=2); ax.axvline(0.70, color="#2563EB", ls="--", lw=2)
    ax.legend(frameon=False, fontsize=11)
    stil(ax, "Model Güveninin Dağılımı", "En yüksek sınıf olasılığı", "Görüntü sayısı")
    fig.tight_layout()
    return fig


def kalibrasyon_grafigi(dogru, guven, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(guven, bins) - 1, 0, n_bins - 1)
    xs, ys, ece = [], [], 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0: continue
        xs.append(guven[m].mean()); ys.append(dogru[m].mean())
        ece += m.mean() * abs(dogru[m].mean() - guven[m].mean())
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Mükemmel kalibrasyon")
    ax.plot(xs, ys, "o-", color="#7C3AED", lw=2.5, ms=8, label=f"Model (ECE = {ece:.3f})")
    ax.legend(frameon=False, fontsize=11); ax.grid(linestyle="--", alpha=0.4)
    stil(ax, "Güvenilirlik (Kalibrasyon) Eğrisi", "Ortalama tahmin güveni", "Gerçek doğruluk")
    fig.tight_layout()
    return fig, ece


def esik_tablosu(dogru, guven):
    kademe = [("Belirsiz (< %40)", guven < 0.40), ("Orta (%40–70)", (guven >= 0.40) & (guven < 0.70)),
              ("Yüksek (≥ %70)", guven >= 0.70)]
    satir = []
    for ad, m in kademe:
        n = int(m.sum())
        satir.append({"Güven kademesi": ad, "Görüntü sayısı": n,
                      "Doğruluk": f"%{dogru[m].mean()*100:.1f}" if n else "-"})
    return pd.DataFrame(satir)


def esik_grafigi(t):
    sayi = t["Görüntü sayısı"].to_numpy()
    dog = [float(str(v).replace("%", "")) if v != "-" else 0 for v in t["Doğruluk"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    b = ax.bar(t["Güven kademesi"], dog, color=["#F59E0B", "#F97316", "#16A34A"], edgecolor="black")
    for r, d, n in zip(b, dog, sayi):
        ax.text(r.get_x() + r.get_width() / 2, d + 1.5, f"%{d:.1f}\n(n={n})", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 118)
    stil(ax, "Güven Kademesine Göre Doğruluk", None, "Doğruluk (%)")
    fig.tight_layout()
    return fig


def veri_dagilimi_grafigi(df):
    ad_k = kolon_bul(df, ["Sınıf Adı"]); cv_k = kolon_bul(df, ["CV"]); te_k = kolon_bul(df, ["Test"])
    kod_k = kolon_bul(df, ["Sınıf Kodu"])
    d = df[df[kod_k].astype(str).str.upper() != "TOPLAM"].copy()
    adlar = d[ad_k].astype(str).str.split(" \\(").str[0].tolist()
    cv = d[cv_k].astype(float).to_numpy(); te = d[te_k].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(adlar, cv, color="#6366F1", edgecolor="black", label="Eğitim / 5-katlı CV")
    ax.bar(adlar, te, bottom=cv, color="#EC4899", edgecolor="black", label="Ayrılmış test")
    for i, (a, b) in enumerate(zip(cv, te)):
        ax.text(i, a + b + 6, f"{int(a + b)}", ha="center", fontweight="bold", fontsize=11)
    ax.legend(frameon=False, fontsize=11)
    stil(ax, "Veri Setinin Sınıflara Dağılımı", None, "Görüntü sayısı")
    fig.tight_layout()
    tablo = pd.DataFrame({"Sınıf": adlar, "Eğitim/CV": cv.astype(int), "Test": te.astype(int),
                          "Toplam": (cv + te).astype(int)})
    return fig, tablo


aktif_paket = load_sampiyon_model()

backbone_adi = aktif_paket["backbone"] if aktif_paket else "ConvNeXt-Tiny"
dogruluk = acc_metni(aktif_paket) if aktif_paket else "%89.30"
cihaz = "GPU / CUDA" if DEVICE.type == "cuda" else "CPU"

st.markdown(f"""
<div class="hero">
    <h1>🩺 DETAY: Deri Tanı ve Ön Triyaj Sistemi</h1>
    <p>Yapay zekâ, deri fotoğrafındaki döküntüye bakıp en benzer hastalığı tahmin eder. 🔍</p>
    <span class="chip">🏅 TÜBİTAK 2204-B</span>
    <span class="chip">🧠 {backbone_adi} + SVM</span>
    <span class="chip">🎯 Test doğruluğu: {dogruluk}</span>
    <span class="chip">⚡ {cihaz}</span>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Canlı Lezyon Analizi & Triyaj", "📈 TÜBİTAK Proje Bulguları ve Grafikler"])

with tab1:
    if aktif_paket is None:
        st.error("Model dosyası (`sampiyon_model.joblib`) bulunamadı! Lütfen model dosyasını ana dizine yükleyin.")
        st.stop()

    col_sol, col_sag = st.columns([1, 1.2], gap="large")

    with col_sol:
        st.markdown("### 📸 1. Fotoğrafını Ekle")
        kaynak = st.radio("Kaynak seç:", ["📁 Bilgisayardan Yükle", "📷 Kamera ile Çek"], horizontal=True)
        dosya = None
        if kaynak == "📁 Bilgisayardan Yükle":
            dosya = st.file_uploader("Deri fotoğrafını buraya bırak:", type=["jpg", "jpeg", "png", "webp"])
        else:
            dosya = st.camera_input("Fotoğraf çek:")
        if dosya is not None:
            img_pil = Image.open(dosya).convert("RGB")
            st.image(img_pil, caption="İncelenen fotoğraf", use_container_width=True)

    with col_sag:
        st.markdown("### 🤖 2. Yapay Zekâ Raporu")
        if dosya is None:
            st.markdown('<div class="bilgi-kutu">👈 <b>Başlamak için</b> sol taraftan bir deri fotoğrafı yükle. '
                        'Yapay zekâ fotoğrafı inceleyip sana bir rapor hazırlayacak!</div>', unsafe_allow_html=True)
        else:
            with st.spinner("🔬 Yapay zekâ fotoğrafı inceliyor..."):
                tf = get_transform()
                t_tensor = tf(img_pil).unsqueeze(0).to(DEVICE)
                backbone = load_backbone(aktif_paket["backbone"])
                with torch.no_grad():
                    feats = backbone(t_tensor).cpu().numpy()
                scaler = aktif_paket["scaler"]
                clf = aktif_paket["clf"]
                feats_scaled = scaler.transform(feats)
                tahmin_idx = int(clf.predict(feats_scaled)[0])
                olasiliklar = clf.predict_proba(feats_scaled)[0]
                class_names = aktif_paket.get("class_names", SINIF_SIRA)
                tahmin_kod = class_names[tahmin_idx]
                bilgi = SINIF_BILGI[tahmin_kod]
                guven = float(olasiliklar[tahmin_idx] * 100)

            sirali = np.argsort(olasiliklar)[::-1]
            ilk, ikinci = int(sirali[0]), int(sirali[1])
            fark = float((olasiliklar[ilk] - olasiliklar[ikinci]) * 100)
            b_ilk = SINIF_BILGI[class_names[ilk]]
            b_ikinci = SINIF_BILGI[class_names[ikinci]]

            if guven < BELIRSIZ_ESIK or fark < MIN_FARK:
                seviye = "belirsiz"
            elif guven < YUKSEK_ESIK:
                seviye = "orta"
            else:
                seviye = "yuksek"

            if seviye == "belirsiz":
                st.markdown(kart_html("🤔", "Emin Olamadım",
                    [f"En olası iki aday: <b>{b_ilk['tr_ad']}</b> ve <b>{b_ikinci['tr_ad']}</b>",
                     f"İlk iki sınıf arasındaki fark sadece {fark:.1f} puan."],
                    [f"Güven: %{guven:.1f}", "🟡 Belirsiz"], "#B45309", "#F59E0B"), unsafe_allow_html=True)
                st.markdown("🩺 **Yönlendirme:** Kesin değerlendirme için bir sağlık uzmanına başvurulmalıdır.")
            else:
                etiket = "🟢 Yüksek güven" if seviye == "yuksek" else "🟠 Orta güven"
                st.markdown(kart_html(bilgi["emoji"], bilgi["tr_ad"], [bilgi["risk"]],
                    [f"Güven: %{guven:.1f}", etiket], bilgi["renk"], bilgi["renk2"]), unsafe_allow_html=True)
                if seviye == "orta":
                    st.markdown(f'<div class="bilgi-kutu">🟠 <b>Orta güven:</b> Model bu sonucu en olası görüyor '
                                f'(%{guven:.1f}) ama ikinci aday <b>{b_ikinci["tr_ad"]}</b> '
                                f'(%{olasiliklar[ikinci]*100:.1f}). Daha net fotoğrafla tekrar deneyebilirsin.</div>',
                                unsafe_allow_html=True)
                st.markdown(f"📋 **Klinik Açıklama:** {bilgi['aciklama']}")
                st.markdown(f"🩺 **Yönlendirme:** {bilgi['oneri']}")
                st.markdown(f"📚 **Klinik Rehber Kaynağı:** *{bilgi['kaynak']}*")
                st.markdown(f'<div class="cocuk-kutu">💡 <b>Basitçe anlatalım:</b> {bilgi["cocuk"]}</div>', unsafe_allow_html=True)

            st.markdown('<div class="uyari-kutu">⚠️ <b>Önemli:</b> Bu sistem tıbbi tanı koymaz. Bir araştırma prototipidir '
                        've yalnızca dört sınıf (su çiçeği, sağlıklı, kızamık, maymun çiçeği) için eğitilmiştir. '
                        'Doktor muayenesinin yerini tutmaz; şüphe durumunda sağlık kuruluşuna başvurun.</div>',
                        unsafe_allow_html=True)

            st.markdown("### 📊 Hangi Hastalığa Ne Kadar Benziyor?")
            bar_html = ""
            for i in sirali:
                b = SINIF_BILGI[class_names[int(i)]]
                p = float(olasiliklar[int(i)] * 100)
                bar_html += f"""<div class="bar-satir"><div class="bar-ad">{b['emoji']} {b['tr_ad']}</div>
                    <div class="bar-dis"><div class="bar-ic" style="width:{max(p, 8):.1f}%;background:linear-gradient(90deg,{b['renk']},{b['renk2']});">
                    %{p:.2f}</div></div></div>"""
            st.markdown(bar_html, unsafe_allow_html=True)

            st.markdown("### 🔥 Yapay Zekâ Nereye Baktı?")
            with st.spinner("Dikkat haritası hesaplanıyor..."):
                harita = occlusion_haritasi(backbone, clf, scaler, t_tensor, ilk)
                harita_buyuk = np.array(Image.fromarray((harita * 255).astype(np.uint8)).resize(img_pil.size, Image.BILINEAR)) / 255.0
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                ax2.imshow(img_pil)
                ax2.imshow(harita_buyuk, cmap="jet", alpha=0.45)
                ax2.axis("off")
                st.pyplot(fig2)
                plt.close(fig2)
            st.markdown('<div class="bilgi-kutu">🔴 <b>Kırmızı</b> bölgeler, yapay zekânın kararında en çok etkili olan yerlerdir. '
                        'Fotoğrafın o kısmını kapatınca tahmin en çok düşer.</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### 📈 Proje Bulguları")
    tahmin_yol = dosya_bul(["*Test_Tahmin*.csv"])
    veri_yol = dosya_bul(["Tablo1*.csv"])

    if tahmin_yol is None:
        st.info("Test tahmin CSV dosyası (*Test_Tahmin*.csv) bulunamadı.")
    else:
        try:
            y, p, olas = tahmin_verisi_hazirla(pd.read_csv(tahmin_yol))
            guven_arr = olas.max(axis=1)
            dogru = (y == p)
            t_metrik = metrik_tablosu(y, p)
            makro_f1 = float(t_metrik["F1"].mean())

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(metrik_kutusu(len(y), "Test görüntüsü"), unsafe_allow_html=True)
            c2.markdown(metrik_kutusu(f"%{dogru.mean()*100:.2f}", "Doğruluk"), unsafe_allow_html=True)
            c3.markdown(metrik_kutusu(f"{makro_f1:.3f}", "Makro F1"), unsafe_allow_html=True)
            c4.markdown(metrik_kutusu(f"{int(dogru.sum())} / {len(y)}", "Doğru tahmin"), unsafe_allow_html=True)
            st.write("")

            if veri_yol is not None:
                st.markdown("#### 1. Veri Seti")
                try:
                    fig_v, tablo_v = veri_dagilimi_grafigi(pd.read_csv(veri_yol))
                    a, b = st.columns([1.3, 1])
                    with a:
                        fig_kaydet_goster(fig_v, "sekil1_veri_dagilimi.png", 1,
                                          "Veri setinin sınıflara göre dağılımı (eğitim/5-katlı CV ve ayrılmış test).")
                    with b:
                        st.markdown("**Tablo 1.** Sınıf başına görüntü sayıları")
                        st.dataframe(tablo_v, hide_index=True, use_container_width=True)
                except Exception as e:
                    st.warning(f"Veri dağılımı grafiği oluşturulamadı: {e}")

            st.markdown("#### 2. Sınıflandırma Başarımı")
            a, b = st.columns(2)
            with a:
                fig_kaydet_goster(cm_grafigi(y, p), "sekil2_karisiklik_matrisi.png", 2,
                                  "Ayrılmış test kümesi için karışıklık matrisi (sayı ve satır yüzdesi).")
            with b:
                fig_kaydet_goster(metrik_grafigi(t_metrik), "sekil3_sinif_basarimi.png", 3,
                                  "Sınıf bazında kesinlik, duyarlılık ve F1 skorları.")
            st.markdown("**Tablo 2.** Sınıf bazında başarım ölçütleri")
            tt = t_metrik.copy()
            for k in ("Kesinlik", "Duyarlılık", "F1"):
                tt[k] = tt[k].map(lambda v: f"{v:.3f}")
            st.dataframe(tt, hide_index=True, use_container_width=True)

            st.markdown("#### 3. Ayırt Edicilik ve Güvenilirlik")
            a, b = st.columns(2)
            with a:
                fig_kaydet_goster(roc_grafigi(y, olas), "sekil4_roc.png", 4,
                                  "Sınıflar için bire-karşı-hepsi ROC eğrileri ve AUC değerleri.")
            with b:
                fig_k, ece = kalibrasyon_grafigi(dogru, guven_arr)
                fig_kaydet_goster(fig_k, "sekil5_kalibrasyon.png", 5,
                                  f"Güvenilirlik eğrisi (ECE = {ece:.3f}). Köşegene yakınlık iyi kalibrasyonu gösterir.")

            st.markdown("#### 4. Güven Eşiği Analizi")
            a, b = st.columns(2)
            with a:
                fig_kaydet_goster(guven_dagilimi_grafigi(dogru, guven_arr), "sekil6_guven_dagilimi.png", 6,
                                  "Doğru ve yanlış tahminlerde model güveninin dağılımı (kesikli çizgiler %40 ve %70 eşikleri).")
            with b:
                t_esik = esik_tablosu(dogru, guven_arr)
                fig_kaydet_goster(esik_grafigi(t_esik), "sekil7_esik_dogruluk.png", 7,
                                  "Güven kademesine göre doğruluk. Uygulamadaki üç kademeli eşik sistemi bu analize dayanır.")
            st.markdown("**Tablo 3.** Güven kademesine göre doğruluk")
            st.dataframe(t_esik, hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"Grafikler oluşturulurken hata: {e}")

    with st.expander("📂 Ham CSV tablolarını göster"):
        csv_listesi = []
        for d in ARAMA_DIZINLERI:
            if os.path.exists(d):
                csv_listesi.extend(glob.glob(os.path.join(d, "*.csv")))
        secilen_adlar = set()
        for yol in sorted(csv_listesi):
            ad = os.path.basename(yol)
            if ad not in secilen_adlar:
                secilen_adlar.add(ad)
                st.markdown(f"**{ad}**")
                st.dataframe(pd.read_csv(yol), use_container_width=True)