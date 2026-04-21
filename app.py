import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Vector Art Studio", layout="wide")

st.title("🚗 Studio de Vectorisation (Version Stable)")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("1. Charger la photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Réglages du tracé")
    threshold_val = st.sidebar.slider("Seuil (Threshold)", 0, 255, 128)
    invert = st.sidebar.toggle("Inverser les couleurs", value=False)
    blur_res = st.sidebar.slider("Lissage", 1, 21, 5, step=2)
    line_thickness = st.sidebar.slider("Épaisseur des traits", -10, 10, 0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Cadre de maintien")
    add_border = st.sidebar.toggle("Ajouter un cadre", value=True)
    border_thickness = st.sidebar.slider("Épaisseur du cadre", 5, 150, 30)

    # --- PIPELINE DE TRAITEMENT ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_res, blur_res), 0)
    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)

    if invert:
        thresh = cv2.bitwise_not(thresh)

    if line_thickness != 0:
        kernel = np.ones((abs(line_thickness), abs(line_thickness)), np.uint8)
        if line_thickness > 0:
            thresh = cv2.erode(thresh, kernel, iterations=1)
        else:
            thresh = cv2.dilate(thresh, kernel, iterations=1)

    if add_border:
        h, w = thresh.shape
        cv2.rectangle(thresh, (0, 0), (w, h), (0, 0, 0), border_thickness)

    # --- AFFICHAGE ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("Rendu Final")
        st.image(thresh, use_container_width=True)

# --- EXPORT SVG (VERSION AMÉLIORÉE) ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 GÉNÉRER LE SVG"):
        # On trouve les contours de l'image
        # RETR_TREE est important pour garder la hiérarchie (trous dans les formes)
        contours, hierarchy = cv2.findContours(cv2.bitwise_not(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = thresh.shape
        # AJOUT de fill-rule="evenodd" pour gérer les trous
        svg_header = f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
        svg_footer = "</svg>"
        svg_paths = []

        for cnt in contours:
            if len(cnt) > 2:
                # Création du tracé
                path_data = "M " + " L ".join([f"{p[0][0]},{p[0][1]}" for p in cnt]) + " Z"
                # On applique fill-rule="evenodd" ici
                svg_paths.append(f'<path d="{path_data}" fill="black" fill-rule="evenodd" stroke="none" />')

        svg_full = svg_header + "".join(svg_paths) + svg_footer
        
        st.sidebar.success("✅ SVG optimisé généré !")
        st.sidebar.download_button(
            label="📥 TÉLÉCHARGER LE SVG",
            data=svg_full,
            file_name="mon_art_metal.svg",
            mime="image/svg+xml"
        )
        )
else:
    st.info("👋 Chargez une photo. Cette version n'utilise plus 'vtracer' pour éviter les erreurs.")
