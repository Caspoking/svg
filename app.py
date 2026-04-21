import streamlit as st
import cv2
import numpy as np
from PIL import Image
import vtracer
import os

# Configuration de la page
st.set_page_config(page_title="Photo to Metal Art", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2e7d32; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎨 Convertisseur d'Art Vectoriel")
st.write("Transformez n'importe quelle photo en design prêt pour la découpe ou le print.")

# --- SIDEBAR : PARAMÈTRES ---
st.sidebar.header("⚙️ Réglages du Design")

uploaded_file = st.sidebar.file_uploader("Étape 1 : Charger une image", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("---")
st.sidebar.subheader("Étape 2 : Ajuster le rendu")

# Paramètres de l'image
threshold_val = st.sidebar.slider("Contraste (Seuil)", 0, 255, 128, help="Ajuste la quantité de noir.")
blur_res = st.sidebar.slider("Lissage des bords", 1, 21, 5, step=2)
invert = st.sidebar.checkbox("Inverser les couleurs", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Étape 3 : Cadre de maintien")
add_border = st.sidebar.toggle("Ajouter un cadre noir", value=True)
border_thickness = st.sidebar.slider("Épaisseur du cadre", 5, 100, 20)

if uploaded_file is not None:
    # Traitement de l'image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Original")
        st.image(img_rgb, use_container_width=True)

    # --- PIPELINE DE TRAITEMENT ---
    # 1. Gris et Flou
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_res, blur_res), 0)
    
    # 2. Threshold (Noir et Blanc)
    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
    
    if invert:
        thresh = cv2.bitwise_not(thresh)

    # 3. Ajout du Cadre (Border)
    if add_border:
        h, w = thresh.shape
        # On dessine un rectangle noir sur les bords
        t = border_thickness
        cv2.rectangle(thresh, (0, 0), (w, h), (0, 0, 0), t)

    with col2:
        st.subheader("✨ Rendu Automatique")
        st.image(thresh, use_container_width=True, channels="GRAY")

    # --- EXPORT VECTORIEL ---
    st.markdown("---")
    if st.button("🚀 GÉNÉRER LE FICHIER VECTORIEL (SVG)"):
        with st.spinner("Vectorisation en cours..."):
            # Sauvegarde temporaire pour vtracer
            temp_png = "temp_render.png"
            cv2.imwrite(temp_png, thresh)
            output_svg = "mon_design.svg"
            
            # vtracer convertit le pixel en courbe
            vtracer.convert_image_to_svg(temp_png, output_svg)

            with open(output_svg, "rb") as f:
                st.download_button(
                    label="📥 TÉLÉCHARGER LE SVG",
                    data=f,
                    file_name="art_vectoriel.svg",
                    mime="image/svg+xml"
                )
            st.success("Vectorisation terminée avec succès !")
else:
    st.info("Veuillez charger une photo dans la barre latérale pour commencer.")
