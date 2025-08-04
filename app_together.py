import streamlit as st
import io
import os
import base64
import re
from PIL import Image
from fpdf import FPDF
from together import Together

# === 🔐 Setup Together.ai API key ===
os.environ["TOGETHER_API_KEY"] = "10c1c7c6fabe12373eee8e5ef785d62396cb7e35c0500e6eab8097f3c5fd2187"
client = Together()

# === 🖼️ Base64 encoding for LLaMA input ===
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# === 🧠 Extract text using LLaMA OCR ===
def extract_text_llama(image_bytes):
    base64_img = encode_image(image_bytes)
    prompt = "Extract the exact text from the image without adding any explanation or description. Return only the raw text string."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]
        }
    ]

    try:
        stream = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=messages,
            stream=True
        )
        result = ""
        for chunk in stream:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    result += delta.content
        return result.strip()
    except Exception as e:
        return f"❌ LLaMA OCR Error: {str(e)}"

# === 📄 Unicode PDF generator ===
class UnicodePDF(FPDF):
    def __init__(self):
        super().__init__()
        font_path = "DejaVuSans.ttf"
        if not os.path.exists(font_path):
            st.error("⚠️ Font not found: fonts/DejaVuSans.ttf")
            st.stop()
        self.add_font("DejaVu", "", font_path, uni=True)
        self.set_font("DejaVu", "", 12)
        self.set_auto_page_break(auto=True, margin=15)

# === 🧮 Accuracy Functions ===
def clean_and_split(text):
    return re.sub(r"[^\w@-]", " ", text.lower()).split()

def compute_accuracy(predicted, ground_truth):
    pred_words = set(clean_and_split(predicted))
    gt_words = set(clean_and_split(ground_truth))
    total = len(gt_words)
    correct = len(pred_words.intersection(gt_words))
    accuracy = round((correct / total) * 100, 2) if total else 0.0
    return accuracy, correct, total

def get_mismatched_words(predicted, ground_truth):
    pred_set = set(clean_and_split(predicted))
    gt_words = clean_and_split(ground_truth)
    return [word for word in gt_words if word not in pred_set]

# === 🚀 Streamlit App ===
st.set_page_config(page_title="📝 LLaMA OCR to PDF", layout="centered")
st.title("📝 Convert Handwritten Text to Digital Text & PDF")

uploaded_files = st.file_uploader(
    "Upload handwritten/printed image(s) (JPEG/PNG, <4MB each):",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Cache OCR results across reruns
if "ocr_results_llama" not in st.session_state:
    st.session_state.ocr_results_llama = {}

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        st.divider()
        st.subheader(f"📄 File {idx+1}: {uploaded_file.name}")

        image_bytes = uploaded_file.read()
        if len(image_bytes) > 4 * 1024 * 1024:
            st.error("❌ Image too large. Must be under 4MB.")
            continue

        st.image(image_bytes, use_column_width=True)

        # OCR if not already done
        if uploaded_file.name not in st.session_state.ocr_results_llama:
            with st.spinner("🧠 Extracting text using LLaMA OCR..."):
                extracted_text = extract_text_llama(image_bytes)
                st.session_state.ocr_results_llama[uploaded_file.name] = extracted_text

        extracted_text = st.session_state.ocr_results_llama[uploaded_file.name]
        st.success("✅ Text Extracted")
        st.text_area("Extracted Text", value=extracted_text, height=200, key=f"ocr_text_{idx}")

        # === 🧾 Ground Truth Accuracy Section ===
        st.subheader("🔎 Check OCR Accuracy (Optional)")
        ground_truth = st.text_area("✍️ Enter ground truth text:", height=200, key=f"gt_input_{idx}")

        if st.button("✅ Check Accuracy", key=f"check_acc_{idx}"):
            if ground_truth:
                accuracy, correct, total = compute_accuracy(extracted_text, ground_truth)
                mismatches = get_mismatched_words(extracted_text, ground_truth)

                st.info(f"📊 Word-Level Accuracy: **{accuracy}%** ({correct}/{total} correct)")
                if mismatches:
                    st.warning("❌ Mismatched words:")
                    st.code(", ".join(mismatches))
            else:
                st.warning("⚠️ Please enter ground truth first.")

        # === 📥 Download PDF Button ===
        pdf = UnicodePDF()
        pdf.add_page()
        pdf.set_font("DejaVu", size=14)
        pdf.cell(0, 10, uploaded_file.name, ln=True)
        pdf.set_font("DejaVu", size=12)
        pdf.multi_cell(0, 10, extracted_text)

        # Convert PDF output to valid bytes
        pdf_content = pdf.output(dest="S")
        pdf_bytes = pdf_content.encode("latin1") if isinstance(pdf_content, str) else bytes(pdf_content)

        st.download_button(
            label="📥 Download as PDF",
            data=pdf_bytes,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_llama.pdf",
            mime="application/pdf",
            key=f"dl_btn_{idx}"
        )
