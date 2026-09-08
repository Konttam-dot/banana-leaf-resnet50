import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(
    page_title="Banana Leaf Classification",
    page_icon="🍌"
)

st.title("🍌 Banana Leaf Classification")
st.write("จำแนกประเภทใบกล้วยด้วย ResNet50")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =================================
# ชื่อ Class
# =================================
# ตรงนี้ต้องเปลี่ยนให้ตรงกับ Dataset ของคุณ
class_names = [
    "Banana Skipper Damage",
    "Black and Yellow Sigatoka",
    "Chewing insect damage on banana leaf",
    "Healthy Banana leaf",
    "Panama Wilt Disease"
]

# =================================
# โหลดโมเดล
# =================================

@st.cache_resource
def load_model():

    model = models.resnet50(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    model.load_state_dict(
        torch.load(
            "best_resnet50_banana.pth",
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model


model = load_model()

# =================================
# Transform
# =================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =================================
# Upload รูป
# =================================

uploaded_file = st.file_uploader(
    "เลือกรูปใบกล้วย",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="รูปที่อัปโหลด",
        use_container_width=True
    )

    if st.button("🔍 ทำนาย"):

        image_tensor = transform(
            image
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            output = model(
                image_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                1
            )

        predicted_class = class_names[
            predicted.item()
        ]

        confidence = (
            confidence.item() * 100
        )

        st.success(
            f"ผลการทำนาย: {predicted_class}"
        )

        st.info(
            f"ความมั่นใจ: {confidence:.2f}%"
        )
