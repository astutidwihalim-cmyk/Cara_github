
import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ============================================================
# PATH MODEL DAN LABEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "model_25class.tflite"
)

LABELS_PATH = os.path.join(
    BASE_DIR,
    "model",
    "labels.txt"
)

IMAGE_SIZE = 128


# ============================================================
# KONFIGURASI WEBSITE
# ============================================================

st.set_page_config(
    page_title="Klasifikasi Komponen Elektronika",
    page_icon="🔌",
    layout="centered"
)

st.title("🔌 Sistem Klasifikasi Komponen Elektronika")

st.write(
    "Upload gambar komponen elektronika untuk mengetahui "
    "hasil klasifikasi menggunakan Machine Learning."
)


# ============================================================
# CEK FILE
# ============================================================

if not os.path.exists(MODEL_PATH):
    st.error("❌ File model tidak ditemukan.")
    st.stop()

if not os.path.exists(LABELS_PATH):
    st.error("❌ File labels.txt tidak ditemukan.")
    st.stop()


# ============================================================
# LOAD LABELS
# ============================================================

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]


# ============================================================
# LOAD MODEL TFLITE
# ============================================================

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )
    interpreter.allocate_tensors()
    return interpreter


interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ============================================================
# UPLOAD GAMBAR
# ============================================================

uploaded_file = st.file_uploader(
    "Pilih gambar komponen",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDIKSI
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang diupload",
        use_container_width=True
    )

    # Resize
    image_resized = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Konversi ke array float32
    image_array = np.array(
        image_resized
    ).astype(np.float32)

    # Tambahkan batch dimension
    input_data = np.expand_dims(
        image_array,
        axis=0
    )

    # Masukkan gambar ke model
    interpreter.set_tensor(
        input_details[0]["index"],
        input_data
    )

    # Jalankan model
    interpreter.invoke()

    # Ambil hasil
    output_data = interpreter.get_tensor(
        output_details[0]["index"]
    )

    probabilities = output_data[0]

    # Prediksi utama
    predicted_index = np.argmax(probabilities)

    predicted_label = labels[predicted_index]

    confidence = (
        probabilities[predicted_index] * 100
    )


    # ========================================================
    # HASIL KLASIFIKASI
    # ========================================================

    st.subheader("Hasil Klasifikasi")

    st.success(
        f"Komponen terdeteksi: "
        f"**{predicted_label.replace('_', ' ').title()}**"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


    # ========================================================
    # 5 PREDIKSI TERATAS
    # ========================================================

    st.subheader("5 Prediksi Teratas")

    top_indices = np.argsort(
        probabilities
    )[::-1][:5]

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        label = labels[index].replace(
            "_", " "
        ).title()

        score = probabilities[index] * 100

        st.write(
            f"{rank}. **{label}** — "
            f"{score:.2f}%"
        )

        st.progress(
            float(probabilities[index])
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Sistem Klasifikasi Komponen Elektronika "
    "Berbasis Web Menggunakan Machine Learning"
)
