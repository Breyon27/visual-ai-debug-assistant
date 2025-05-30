
import streamlit as st
import torch
import open_clip
import numpy as np
from PIL import Image
import imageio
import csv
import tempfile
import os
import base64
from openai import openai
from sklearn.metrics.pairwise import cosine_similarity

# Config
EMBEDDING_FILE = "clip_embeddings.csv"
FIX_LOG_FILE = "fix_log.csv"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

# OpenAI key
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Load model
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device)
    return model, preprocess, device

def embed_image(image, model, preprocess, device):
    image = image.convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_tensor).squeeze(0).cpu().numpy()
    return embedding

def load_embeddings():
    embeddings, filenames = [], []
    with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            filenames.append(row[0])
            embeddings.append(np.array(row[1:], dtype=np.float32))
    return filenames, np.stack(embeddings)

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

def extract_frames_from_video(video_path, interval=2):
    reader = imageio.get_reader(video_path, 'ffmpeg')
    fps = reader.get_meta_data()['fps']
    frame_interval = int(fps * interval)
    frames = []
    for i, frame in enumerate(reader):
        if i % frame_interval == 0:
            image = Image.fromarray(frame)
            frames.append(image)
    reader.close()
    return frames

def generate_ai_fix(image: Image.Image, question: str, frame_name="unknown"):
    buffered = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(buffered.name)
    with open(buffered.name, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")

    try:
        response = client.chat.completions.create(

            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Unity game development expert who helps debug gameplay based on screenshots."},
                {"role": "user", "content": f"This is a screenshot from a Unity game.\n\nQuestion: {question or 'What is the issue here?'}"},
                {"role": "user", "content": {"image": image_data, "mime_type": "image/jpeg"}}
            ],
            max_tokens=500
        )
        fix = response.choices[0].message.content.strip()
        log_ai_fix(frame_name, question, fix)
        return fix
    except Exception as e:
        return f"Error generating AI fix: {e}"

# Streamlit UI
st.title("🎮 Visual AI Debug Assistant")
st.markdown("Upload a gameplay **frame** or **video**. We’ll show a fix from history or generate one using GPT-4o.")

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
        extracted_images = extract_frames_from_video(video_path)

        results = []
        for img in extracted_images:
            emb = embed_image(img, model, preprocess, device).reshape(1, -1)
            similarities = cosine_similarity(emb, vectors)[0]
            top_index = np.argmax(similarities)
            match_frame = filenames[top_index]
            score = similarities[top_index]
            results.append((match_frame, score, fix_log.get(match_frame, {}), img))

        results.sort(key=lambda x: -x[1])

        st.subheader("🧠 Top Matching Fixes from Video:")
        for match_frame, score, fix_data, preview in results[:3]:
            st.image(preview, caption=f"Matched: {match_frame} (Score: {score:.4f})", use_column_width=True)
            if fix_data:
                st.markdown(f"**Prompt:** {fix_data['prompt']}")
                st.success(f"Fix: {fix_data['fix']}")
            else:
                st.warning("No fix found. Trying GPT-4o...")
                gpt_fix = generate_ai_fix(preview, question)
                st.info(f"🧠 AI Fix:\n\n{gpt_fix}")
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
            st.warning("No fix found. Trying GPT-4o...")
            gpt_fix = generate_ai_fix(image, question, frame_name="uploaded_frame.jpg")
            gpt_fix = generate_ai_fix(preview, question, frame_name=f"video_frame_{idx+1}.jpg")
            st.info(f"🧠 AI Fix:\n\n{gpt_fix}")
