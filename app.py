import streamlit as st
import cv2
import numpy as np
from PIL import Image
import vtracer
import tempfile
import os

st.set_page_config(page_title="Vector Art Studio", layout="wide")

st.title("🚗 Studio de Vectorisation Pro")

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

    # --- EXPORT ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 GÉNÉRER LE SVG"):
        with tempfile.TemporaryDirectory() as tmpdirname:
            input_path = os.path.join(tmpdirname, "input.png")
            output_path = os.path.join(tmpdirname, "output.svg")
            
            # Sauvegarde de l'image traitée en PNG
            cv2.imwrite(input_path, thresh)
            
            try:
                # LA MÉTHODE LA PLUS STABLE POUR VTRACER PYTHON
                vtracer.vtrace(
                    input_path,
                    output_path,
                    mode="spline",      # Pour avoir de belles courbes
                    limit_ratio=10,     # Niveau de détail
                    filter_speckle=4,   # Supprime les petits points parasites
                    color_precision=6   # Précision des couleurs (ici N&B)
                )
                
                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        svg_data = f.read()
                    
                    st.sidebar.success("✅ SVG Prêt !")
                    st.sidebar.download_button(
                        label="📥 TÉLÉCHARGER LE SVG",
                        data=svg_data,
                        file_name="mon_design_vector.svg",
                        mime="image/svg+xml"
                    )
                else:
                    st.error("Le fichier SVG n'a pas pu être généré.")
                    
            except Exception as e:
                st.error(f"Erreur technique : {e}")
                st.info("Astuce : Si l'erreur persiste, vérifiez que votre fichier requirements.txt contient bien 'vtracer'.")
else:
    st.info("👋 Chargez une photo pour commencer.")
