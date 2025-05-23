
import streamlit as st
import torch
import open_clip
import numpy as np
from PIL import Image
import csv
from sklearn.metrics.pairwise import cosine_similarity
import os

# === Config ===
EMBEDDING_FILE = "clip_embeddings.csv"
FIX_LOG_FILE = "fix_log.csv"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

# === Load model ===
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device)
    return model, preprocess, device

# === Embed image ===
def embed_image(image, model, preprocess, device):
    image = image.convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_tensor).squeeze(0).cpu().numpy()
    return embedding

# === Load embeddings from CSV ===
def load_embeddings():
    embeddings, filenames = [], []
    with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            filenames.append(row[0])
            embeddings.append(np.array(row[1:], dtype=np.float32))
    return filenames, np.stack(embeddings)

# === Load fix log from CSV ===
def load_fix_log():
    fixes = {}
    with open(FIX_LOG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fixes[row["frame"]] = {
                "prompt": row["prompt"],
                "fix": row["fix"]
            }
    return fixes

# === UI ===
st.title("🎮 Visual AI Debug Assistant")
st.markdown("Upload a gameplay frame and get the most similar bug fix from history.")

uploaded_image = st.file_uploader("Upload Frame (.jpg or .png)", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Frame", use_column_width=True)

    model, preprocess, device = load_model()
    user_embedding = embed_image(image, model, preprocess, device).reshape(1, -1)
    filenames, vectors = load_embeddings()
    fix_log = load_fix_log()

    similarities = cosine_similarity(user_embedding, vectors)[0]
    top_index = np.argmax(similarities)
    best_match = filenames[top_index]
    score = similarities[top_index]

    st.subheader("🔍 Most Similar Frame:")
    st.write(f"**{best_match}** (Similarity: {score:.4f})")

    if best_match in fix_log:
        st.markdown("### 🧠 Prompt")
        st.write(fix_log[best_match]["prompt"])
        st.markdown("### 🔧 Suggested Fix")
        st.success(fix_log[best_match]["fix"])
    else:
        st.warning("No fix found for this frame.")
