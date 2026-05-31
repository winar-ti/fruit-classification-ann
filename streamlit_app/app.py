import streamlit as st
import numpy as np
import pandas as pd
import cv2
import json

from PIL import Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Fruit Classification ANN",
    page_icon="🍎",
    layout="wide"
)

# =====================================================
# LOAD CLASSES
# =====================================================

try:

    with open(
        "classes.json",
        "r"
    ) as f:

        classes = json.load(f)

except FileNotFoundError:

    st.error(
        "❌ File 'classes.json' tidak ditemukan. "
        "Pastikan file ada di direktori yang sama dengan app.py"
    )

    st.stop()

# =====================================================
# CONSTANT
# =====================================================

IMG_SIZE = 64
INPUT_DIM = IMG_SIZE * IMG_SIZE * 3

# =====================================================
# BUILD MODEL
# =====================================================

@st.cache_resource
def load_ai_model():

    model = Sequential([

        Dense(
            256,
            input_shape=(INPUT_DIM,)
        ),

        ReLU(),

        Dense(128),

        ReLU(),

        Dense(64),

        ReLU(),

        Dense(
            len(classes),
            activation="softmax"
        )
    ])

    model.load_weights(
        "fruit_best_weights.weights.h5"
    )

    return model


try:

    model = load_ai_model()

except Exception as e:

    st.error(
        f"❌ Gagal memuat model: {e}\n\n"
        "Pastikan file 'fruit_best_weights.weights.h5' "
        "ada di direktori yang sama dengan app.py"
    )

    st.stop()

# =====================================================
# PREPROCESS IMAGE
# =====================================================

def preprocess_image(image):

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = image.astype(
        "float32"
    ) / 255.0

    image = image.flatten()

    image = np.expand_dims(
        image,
        axis=0
    )

    return image

# =====================================================
# HEADER
# =====================================================

st.title("🍎 Fruit Classification Using ANN")

st.markdown(
    """
### Artificial Neural Network Classification

Dataset : Fruit360

Activation Function Comparison:

- ReLU
- Leaky ReLU

🏆 Winner Model : ReLU
"""
)

# =====================================================
# SIDEBAR
# =====================================================

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Prediction",
        "Model Information"
    ]
)

# =====================================================
# HOME
# =====================================================

if menu == "Home":

    st.header("Welcome")

    st.write(
        """
This application classifies fruit images
using Artificial Neural Networks.

The experiment compares:

• ReLU

• Leaky ReLU

Winner based on testing accuracy:
ReLU
"""
    )

    st.success(
        "🏆 Best Model : ReLU"
    )

# =====================================================
# MODEL INFORMATION
# =====================================================

elif menu == "Model Information":

    st.header(
        "Model Information"
    )

    st.subheader(
        "Architecture"
    )

    st.code(
        """
Input Layer : 12288

Dense(256)
ReLU

Dense(128)
ReLU

Dense(64)
ReLU

Dense(10)
Softmax
"""
    )

    st.subheader(
        "Experiment Result"
    )

    st.table({

        "Model": [
            "ReLU",
            "Leaky ReLU"
        ],

        "Accuracy": [
            "98.34%",
            "97.92%"
        ],

        "Loss": [
            "0.054951",
            "0.039673"
        ]
    })

    st.subheader(
        "Fruit Classes"
    )

    for fruit in classes:

        st.write(
            f"• {fruit}"
        )

# =====================================================
# PREDICTION
# =====================================================

elif menu == "Prediction":

    st.header(
        "Fruit Prediction"
    )

    uploaded_file = st.file_uploader(
        "Upload Fruit Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        image = image.convert(
            "RGB"
        )

        image_np = np.array(
            image
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with st.spinner("Sedang memproses prediksi..."):

            try:

                processed = preprocess_image(
                    image_np
                )

                prediction = model.predict(
                    processed,
                    verbose=0
                )

                predicted_class = np.argmax(
                    prediction
                )

                confidence = (
                    prediction[0][predicted_class]
                    * 100
                )

                with col2:

                    st.success(
                        f"Prediction : {classes[predicted_class]}"
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                    # Top 3 Predictions
                    st.markdown("**Top 3 Predictions:**")

                    top3_idx = np.argsort(
                        prediction[0]
                    )[::-1][:3]

                    medals = ["🥇", "🥈", "🥉"]

                    for rank, idx in enumerate(top3_idx):

                        prob_pct = prediction[0][idx] * 100

                        st.write(
                            f"{medals[rank]} "
                            f"{classes[idx]} "
                            f"— {prob_pct:.2f}%"
                        )

                st.subheader(
                    "Prediction Probability"
                )

                prob_dict = {}

                for i, fruit in enumerate(classes):

                    prob_dict[fruit] = float(
                        prediction[0][i]
                    ) * 100

                prob_df = pd.DataFrame.from_dict(
                    {"Probability (%)": prob_dict},
                    orient="columns"
                )

                st.bar_chart(prob_df)

            except Exception as e:

                st.error(
                    f"❌ Terjadi error saat prediksi: {e}"
                )
