import streamlit as st
import cv2
import numpy as np
from PIL import Image
import vtracer
import os

st.set_page_config(page_title="Vector Art Studio", layout="wide")

st.title("🚗 Studio de Vectorisation Pro")

# --- SIDEBAR : PARAMÈTRES ---
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("1. Charger la photo", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("---")
st.sidebar.subheader("2. Réglages du tracé")
threshold_val = st.sidebar.slider("Seuil (Noir/Blanc)", 0, 255, 128)
blur_res = st.sidebar.slider("Lissage des courbes", 1, 21, 5, step=2)

# NOUVELLE OPTION : ÉPAISSEUR DES TRAITS
line_thickness = st.sidebar.slider("Épaisseur des traits (Détails)", -10, 10, 0, help="Négatif pour affiner, positif pour épaissir.")

st.sidebar.markdown("---")
st.sidebar.subheader("3. Cadre de maintien")
add_border = st.sidebar.toggle("Ajouter un cadre", value=True)
border_thickness = st.sidebar.slider("Épaisseur du cadre", 5, 150, 30)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # --- PIPELINE DE TRAITEMENT ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_res, blur_res), 0)
    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)

    # APPLICATION DE L'ÉPAISSEUR (MORPHOLOGIE)
    if line_thickness != 0:
        kernel = np.ones((abs(line_thickness), abs(line_thickness)), np.uint8)
        if line_thickness > 0:
            # Éroder le blanc = épaissir le noir
            thresh = cv2.erode(thresh, kernel, iterations=1)
        else:
            # Dilater le blanc = affiner le noir
            thresh = cv2.dilate(thresh, kernel, iterations=1)

    # AJOUT DU CADRE
    if add_border:
        h, w = thresh.shape
        cv2.rectangle(thresh, (0, 0), (w, h), (0, 0, 0), border_thickness)

    # PRÉPARATION DE LA PRÉVIEW (Ajout d'une bordure blanche pour la visibilité)
    # On ajoute 10 pixels de blanc tout autour pour voir le cadre noir si le fond du site est noir
    preview_img = cv2.copyMakeBorder(thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("Rendu (Aperçu sur fond blanc)")
        st.image(preview_img, use_container_width=True)

    # --- EXPORT ---
    if st.button("🚀 GÉNÉRER LE SVG"):
        temp_png = "render.png"
        cv2.imwrite(temp_png, thresh)
        vtracer.convert_image_to_svg(temp_png, "output.svg")
        
        with open("output.svg", "rb") as f:
            st.download_button("📥 Télécharger le Vectoriel", f, "mon_design.svg", "image/svg+xml")
