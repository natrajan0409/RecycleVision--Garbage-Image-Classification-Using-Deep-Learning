import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RecycleVision",
    page_icon="♻️",
    layout="centered"
)

# ─── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'battery', 'biological', 'brown-glass', 'cardboard',
    'clothes', 'green-glass', 'metal', 'paper',
    'plastic', 'shoes', 'trash', 'white-glass'
]

CLASS_EMOJIS = {
    'battery': '🔋', 'biological': '🌿', 'brown-glass': '🟤',
    'cardboard': '📦', 'clothes': '👕', 'green-glass': '🟢',
    'metal': '🔩', 'paper': '📄', 'plastic': '🧴',
    'shoes': '👟', 'trash': '🗑️', 'white-glass': '⚪'
}

RECYCLING_TIPS = {
    'battery':      "⚠️ Do NOT put in regular trash! Take to a battery recycling drop-off.",
    'biological':   "🌱 Compost it! Great for garden or food waste bins.",
    'brown-glass':  "♻️ Place in glass recycling bin. Remove lids first.",
    'cardboard':    "📦 Flatten and place in paper/cardboard recycling bin.",
    'clothes':      "👕 Donate if usable, or drop at textile collection points.",
    'green-glass':  "♻️ Place in glass recycling bin. Rinse before recycling.",
    'metal':        "🔩 Place in metal/can recycling bin. Rinse cans first.",
    'paper':        "📄 Recycle in paper bin. Keep dry and clean.",
    'plastic':      "🧴 Check the recycling number. Most plastics go in recycling bin.",
    'shoes':        "👟 Donate usable shoes. Otherwise use textile recycling.",
    'trash':        "🗑️ General waste bin. Cannot be recycled.",
    'white-glass':  "♻️ Place in glass recycling bin. Remove metal lids."
}

MODEL_PATH = "models/best_model.pth"

# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 12)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

# ─── Preprocess Image ──────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ─── Predict ───────────────────────────────────────────────────────────────────
def predict(model, tensor: torch.Tensor):
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
    top3_probs, top3_idx = torch.topk(probs, 3)
    return (
        CLASS_NAMES[top3_idx[0]],
        float(top3_probs[0]) * 100,
        [(CLASS_NAMES[idx], float(p) * 100) for idx, p in zip(top3_idx, top3_probs)]
    )

# ─── UI ────────────────────────────────────────────────────────────────────────
st.title("♻️ RecycleVision")
st.subheader("Garbage Image Classification Using Deep Learning")
st.markdown("Upload an image of garbage and let AI classify it for you!")
st.divider()

uploaded_file = st.file_uploader(
    "📸 Upload a garbage image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("🔍 Analyzing image..."):
            try:
                model = load_model()
                tensor = preprocess_image(image)
                pred_class, confidence, top3 = predict(model, tensor)

                emoji = CLASS_EMOJIS.get(pred_class, "♻️")
                tip   = RECYCLING_TIPS.get(pred_class, "")

                st.success(f"### {emoji} {pred_class.upper()}")
                st.metric("Confidence", f"{confidence:.2f}%")
                st.info(tip)

                st.markdown("#### 🏆 Top 3 Predictions")
                for cls, prob in top3:
                    st.progress(int(prob), text=f"{CLASS_EMOJIS.get(cls,'')} {cls}: {prob:.1f}%")

            except FileNotFoundError:
                st.error("❌ Model not found! Please place `best_model.pth` in the `models/` folder.")

else:
    st.info("👆 Upload an image to get started!")
    st.markdown("""
    **Supported categories:**
    🔋 Battery | 🌿 Biological | 🟤 Brown Glass | 📦 Cardboard  
    👕 Clothes | 🟢 Green Glass | 🔩 Metal | 📄 Paper  
    🧴 Plastic | 👟 Shoes | 🗑️ Trash | ⚪ White Glass
    """)

st.divider()
st.caption("RecycleVision | EfficientNetB0 | Accuracy: 95.42% | Built with PyTorch & Streamlit")
