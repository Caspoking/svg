import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Vector Art Studio", layout="wide")

st.title("🚗 Studio de Vectorisation avec Ajustement")

# --- SIDEBAR ---
st.sidebar.header("⚙️ 1. Image & Tracé")
uploaded_file = st.sidebar.file_uploader("Charger la photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_orig = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h_orig, w_orig = img_orig.shape[:2]
    
    threshold_val = st.sidebar.slider("Seuil (Threshold)", 0, 255, 128)
    invert = st.sidebar.toggle("Inverser les couleurs", value=False)
    blur_res = st.sidebar.slider("Lissage", 1, 21, 5, step=2)
    line_thickness = st.sidebar.slider("Épaisseur des traits", -10, 10, 0)

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 2. Positionnement & Zoom")
    zoom = st.sidebar.slider("Zoom", 0.1, 3.0, 1.0, 0.1)
    offset_x = st.sidebar.slider("Déplacement Horizontal (X)", -w_orig, w_orig, 0)
    offset_y = st.sidebar.slider("Déplacement Vertical (Y)", -h_orig, h_orig, 0)

    st.sidebar.markdown("---")
    st.sidebar.header("🖼️ 3. Cadre")
    add_border = st.sidebar.toggle("Ajouter un cadre", value=True)
    border_thickness = st.sidebar.slider("Épaisseur du cadre", 5, 150, 30)

    # --- PIPELINE DE TRANSFORMATION (ZOOM & POSITION) ---
    # Création d'une toile blanche à la taille d'origine
    canvas = np.full((h_orig, w_orig), 255, dtype=np.uint8)
    
    # Préparation de l'image (Gris + Flou + Seuil)
    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_res, blur_res), 0)
    _, binary = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
    
    if invert:
        binary = cv2.bitwise_not(binary)

    # Calcul de la matrice de transformation (Zoom + Déplacement)
    center_x, center_y = w_orig / 2, h_orig / 2
    M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom)
    M[0, 2] += offset_x
    M[1, 2] += offset_y

    # Application de la transformation
    transformed = cv2.warpAffine(binary, M, (w_orig, h_orig), borderValue=(255 if not invert else 0))

    # Épaisseur des traits
    thresh = transformed
    if line_thickness != 0:
        kernel = np.ones((abs(line_thickness), abs(line_thickness)), np.uint8)
        thresh = cv2.erode(thresh, kernel) if line_thickness > 0 else cv2.dilate(thresh, kernel)

    # Ajout du Cadre
    if add_border:
        cv2.rectangle(thresh, (0, 0), (w_orig, h_orig), (0, 0, 0), border_thickness)

    # --- AFFICHAGE ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("Rendu Final")
        st.image(thresh, use_container_width=True)

    # --- EXPORT SVG ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 GÉNÉRER LE SVG"):
        contours, _ = cv2.findContours(cv2.bitwise_not(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        svg_header = f'<svg width="{w_orig}" height="{h_orig}" viewBox="0 0 {w_orig} {h_orig}" xmlns="http://www.w3.org/2000/svg">'
        combined_d = ""
        for cnt in contours:
            if len(cnt) > 2:
                combined_d += "M " + " L ".join([f"{p[0][0]},{p[0][1]}" for p in cnt]) + " Z "
        
        svg_full = f'{svg_header}<path d="{combined_d}" fill="black" fill-rule="evenodd" stroke="none" /></svg>'
        
        st.sidebar.success("✅ SVG Prêt !")
        st.sidebar.download_button("📥 TÉLÉCHARGER LE SVG", data=svg_full, file_name="mon_art_ajuste.svg", mime="image/svg+xml")
else:
    st.info("👋 Chargez une photo pour commencer.")
