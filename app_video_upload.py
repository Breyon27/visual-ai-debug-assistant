
import streamlit as st
import torch
import open_clip
import numpy as np
from PIL import Image
import cv2
import csv
import tempfile
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

# === Extract frames from video ===
def extract_frames_from_video(video_path, interval=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frame_count = 0
    images = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(frame_rgb))
        frame_count += 1

    cap.release()
    return images

# === UI ===
st.title("🎮 Visual AI Debug Assistant")
st.markdown("Upload a gameplay **frame** or **video**, and get AI-powered bug fixes based on your dev history.")

question = st.text_input("Optional: What issue are you trying to fix?")

uploaded_file = st.file_uploader("Upload Frame (.jpg/.png) or Video (.mp4)", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    model, preprocess, device = load_model()
    filenames, vectors = load_embeddings()
    fix_log = load_fix_log()

    if uploaded_file.name.endswith(".mp4"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(uploaded_file.read())
            video_path = tmpfile.name

        st.video(video_path)
        st.write("🔍 Extracting frames from video...")
        extracted_images = extract_frames_from_video(video_path, interval=2)

        results = []
        for idx, img in enumerate(extracted_images):
            emb = embed_image(img, model, preprocess, device).reshape(1, -1)
            similarities = cosine_similarity(emb, vectors)[0]
            top_index = np.argmax(similarities)
            match_frame = filenames[top_index]
            score = similarities[top_index]
            results.append((match_frame, score, fix_log.get(match_frame, {}), img))

        results.sort(key=lambda x: -x[1])  # sort by similarity

        st.subheader("🧠 Best Matching Fixes From Video:")
        for match_frame, score, fix_data, preview in results[:3]:
            st.image(preview, caption=f"Matched: {match_frame} (Score: {score:.4f})", use_column_width=True)
            if fix_data:
                st.markdown(f"**Prompt:** {fix_data['prompt']}")
                st.success(f"Fix: {fix_data['fix']}")
            else:
                st.warning("No fix found for this frame.")
            st.markdown("---")

    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Frame", use_column_width=True)
        emb = embed_image(image, model, preprocess, device).reshape(1, -1)
        similarities = cosine_similarity(emb, vectors)[0]
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
