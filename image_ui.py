"""
image_ui.py
- Streamlit front-end:
  * market_text (Japanese allowed) input
  * auto-translate option (uses OpenAI to translate to English) — optional
  * generation params (width/height/num)
  * generate -> preview -> evaluate -> select for "sale"
  * save chosen images and evaluation CSV, append learning JSON
"""

import os
import io
import streamlit as st
from datetime import datetime
from image_generator import init_pipeline, generate_images, save_image, validate_size
from image_evaluator import evaluate_image_with_ai, append_learning, save_results_csv
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("TNSYSTEM1")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

st.set_page_config(page_title="Image Generation (Complete)", page_icon="🖼️", layout="centered")
st.title("📸 Image Generation — Stable Diffusion + AI Evaluation")

# UI inputs
market_text = st.text_area("Paste market research (Japanese OK). The app can translate below.", height=140)
auto_translate = st.checkbox("Auto-translate (Japanese → English) using OpenAI", value=True)

# size & generation
col1, col2 = st.columns(2)
with col1:
    width = st.number_input("Width (px)", min_value=1000, max_value=8000, value=4000, step=100)
with col2:
    height = st.number_input("Height (px)", min_value=1000, max_value=8000, value=3000, step=100)

num_images = st.slider("Number of images", 1, 5, 3)
filename_base = st.text_input("Base filename (no extension)", value=f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# translate helper
def translate_to_en(text: str) -> str:
    if not client:
        return text
    prompt = (
        "Translate the following Japanese market research text into concise, natural English suitable as an image generation prompt. Keep it descriptive and include keywords.\n\n"
        + text
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a professional translator for image-generation prompts."},
            {"role":"user","content":prompt}
        ],
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()

# Validate size
valid, msg = validate_size(width, height)
if not valid:
    st.error(msg)

# Initialize pipeline lazily
if "pipeline" not in st.session_state:
    try:
        device = "cuda" if (torch := __import__("torch")).cuda.is_available() else "cpu"
        st.session_state.pipeline = init_pipeline(device=device)
    except Exception as e:
        st.warning("Stable Diffusion pipeline initialization failed or is slow. You can still use mock generation if needed.")
        st.session_state.pipeline = None

prompt_en = None
if market_text.strip():
    if auto_translate:
        try:
            prompt_en = translate_to_en(market_text)
            st.markdown("**Translated prompt (EN)**")
            st.write(prompt_en)
        except Exception as e:
            st.warning("Auto-translate failed; using original text as prompt.")
            prompt_en = market_text
    else:
        st.info("Auto-translate disabled; please provide English prompt below if desired.")
        prompt_en = st.text_area("English prompt (optional)", "")

# Generate button
generated_paths = []
generated_images = []
if st.button("Generate images"):
    if not prompt_en or not prompt_en.strip():
        st.error("No prompt provided.")
    elif not valid:
        st.error("Image size invalid for Adobe Stock (4MP minimum).")
    else:
        pipe = st.session_state.pipeline
        if pipe is None:
            st.error("Pipeline not initialized (no GPU/torch or heavy download).")
        else:
            try:
                with st.spinner("Generating..."):
                    imgs = generate_images(pipe, prompt_en, width, height, num_images=num_images)
                st.success("Generation done.")
                for idx, img in enumerate(imgs, start=1):
                    fname_base_i = f"{filename_base}_{idx}"
                    path = save_image(img, fname_base_i)
                    generated_paths.append(path)
                    generated_images.append((path, img))
                    st.image(path, caption=fname_base_i, use_container_width=True)
            except Exception as e:
                st.error(f"Generation failed: {e}")

# Evaluate & Save
if generated_paths:
    if st.button("Evaluate & Save results"):
        rows = []
        for path, img in generated_images:
            scores, comment, sell_score = evaluate_image_with_ai(path, prompt_en)
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": path,
                "prompt": prompt_en,
                "quality": scores.get("quality"),
                "color": scores.get("color"),
                "originality": scores.get("originality"),
                "impact": scores.get("impact"),
                "memorability": scores.get("memorability"),
                "sellability": sell_score,
                "comment": comment
            }
            rows.append(row)
            append_learning({
                "timestamp": row["timestamp"],
                "image_path": path,
                "prompt": prompt_en,
                "scores": scores,
                "sellability": sell_score,
                "comment": comment
            })
        csv_path = save_results_csv(rows)
        st.success(f"Evaluations saved to CSV: {csv_path}")
        st.dataframe(rows)
