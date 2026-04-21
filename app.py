import streamlit as st
import cv2
import numpy as np
from PIL import Image
import vtracer

st.set_page_config(page_title="Vector Art Studio", layout="wide")

# Forcer un style plus clair pour la zone d'aperçu
st.markdown("""
    <style>
    .preview-container {
        background-color: #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚗 Studio de Vectorisation")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("1. Charger la photo", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("---")
st.sidebar.subheader("2. Réglages du tracé")
threshold_val = st.sidebar.slider("Seuil (Threshold)", 0, 255, 128, help="Baissez cette valeur si l'image est trop noire.")
invert = st.sidebar.toggle("Inverser les couleurs (Noir <-> Blanc)", value=False)
blur_res = st.sidebar.slider("Lissage", 1, 21, 5, step=2)
line_thickness = st.sidebar.slider("Épaisseur des traits", -10, 10, 0)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Cadre de maintien")
add_border = st.sidebar.toggle("Ajouter un cadre", value=True)
border_thickness = st.sidebar.slider("Épaisseur du cadre", 5, 150, 30)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # --- PIPELINE ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_res, blur_res), 0)
    
    # Application du seuil
    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)

    # Inversion si demandée (C'est ici qu'on retrouve le fond blanc !)
    if invert:
        thresh = cv2.bitwise_not(thresh)

    # Épaisseur des traits
    if line_thickness != 0:
        kernel = np.ones((abs(line_thickness), abs(line_thickness)), np.uint8)
        thresh = cv2.erode(thresh, kernel) if line_thickness > 0 else cv2.dilate(thresh, kernel)

    # Ajout du Cadre
    if add_border:
        h, w = thresh.shape
        cv2.rectangle(thresh, (0, 0), (w, h), (0, 0, 0), border_thickness)

    # Mise en page
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col2:
        st.subheader("Rendu Final")
        # On affiche sur un fond gris pour bien voir les limites du cadre noir et du fond blanc
        st.image(thresh, caption="Aperçu (Le blanc sera vide, le noir sera découpé)", use_container_width=True)

    # --- BOUTON EXPORT ---
    if st.button("🚀 GÉNÉRER LE SVG"):
        cv2.imwrite("render.png", thresh)
        vtracer.convert_image_to_svg("render.png", "output.svg")
        with open("output.svg", "rb") as f:
            st.download_button("📥 Télécharger le fichier Vectoriel", f, "mon_design.svg", "image/svg+xml")

else:
    st.info("💡 Chargez une image pour commencer. Si le rendu est tout noir, baissez le 'Seuil' ou essayez d'activer 'Inverser les couleurs'.")
