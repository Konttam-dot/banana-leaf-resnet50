import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


# =========================================================
# Page
# =========================================================

st.set_page_config(
    page_title="Banana Leaf AI",
    page_icon="🍌",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f7faf8;
}

.hero {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(135deg, #e8f8ef, #f5fbf7);
    border: 1px solid #d8eee0;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 40px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    color: #60756a;
}

.result-box {
    padding: 25px;
    border-radius: 20px;
    background: white;
    border: 1px solid #e1ebe5;
    margin-top: 20px;
}

.info-box {
    padding: 20px;
    border-radius: 16px;
    background: #ffffff;
    border: 1px solid #e1ebe5;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# Disease Information
# =========================================================

DISEASE_INFO = {

    "Banana Skipper Damage": {
        "thai": "ความเสียหายจากหนอนม้วนใบกล้วย",
        "description": "ใบกล้วยได้รับความเสียหายจาก Banana Skipper",
        "advice": "ตรวจดูใบกล้วยเป็นประจำและจัดการแมลงที่พบอย่างเหมาะสม"
    },

    "Black and Yellow Sigatoka": {
        "thai": "โรคใบจุดดำและเหลือง",
        "description": "โรคที่ทำให้เกิดจุดหรือแผลบนใบกล้วย",
        "advice": "ตัดและจัดการใบที่เป็นโรค ลดความชื้นในแปลง และติดตามอาการของใบ"
    },

    "Chewing insect damage on banana leaf": {
        "thai": "ความเสียหายจากแมลงกัดกินใบ",
        "description": "ใบกล้วยมีร่องรอยจากแมลงกัดกิน",
        "advice": "ตรวจสอบบริเวณใบและต้นกล้วยเป็นประจำ และจัดการแมลงอย่างเหมาะสม"
    },

    "Healthy Banana leaf": {
        "thai": "ใบกล้วยปกติ",
        "description": "ไม่พบลักษณะความเสียหายหรือโรคตามกลุ่มที่โมเดลจำแนก",
        "advice": "ดูแลน้ำ ปุ๋ย และสภาพแวดล้อมของต้นกล้วยอย่างเหมาะสม พร้อมตรวจใบเป็นประจำ"
    },

    "Panama Wilt Disease": {
        "thai": "โรคเหี่ยวปานามา",
        "description": "โรคเหี่ยวปานามา ซึ่งเป็นโรคสำคัญของกล้วย",
        "advice": "หากพบอาการผิดปกติหลายต้น ควรแยกพื้นที่ที่สงสัยและปรึกษาเจ้าหน้าที่เกษตรหรือผู้เชี่ยวชาญ"
    }
}


# =========================================================
# Load Model
# =========================================================

@st.cache_resource
def load_model():

    device = torch.device("cpu")

    # ค้นหาไฟล์โมเดล
    possible_paths = [
        "best_mobilenetv2_banana.pth",
        "mobilenetv2_banana_final.pth",
        "models/best_mobilenetv2_banana.pth",
        "models/mobilenetv2_banana_final.pth"
    ]

    model_path = None

    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        raise FileNotFoundError(
            "ไม่พบไฟล์ MobileNetV2 (.pth) "
            "กรุณาอัปโหลด best_mobilenetv2_banana.pth "
            "ไว้ใน GitHub Repository"
        )

    # โหลด checkpoint
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    # อ่านชื่อคลาสจาก checkpoint
    if isinstance(checkpoint, dict) and "class_names" in checkpoint:

        class_names = checkpoint["class_names"]

    else:

        class_names = [
            "Banana Skipper Damage",
            "Black and Yellow Sigatoka",
            "Chewing insect damage on banana leaf",
            "Healthy Banana leaf",
            "Panama Wilt Disease"
        ]

    # จำนวนคลาส
    num_classes = len(class_names)

    # สร้าง MobileNetV2
    model = models.mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    # อ่าน state_dict
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # รองรับ DataParallel
    new_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        new_state_dict[key] = value

    model.load_state_dict(
        new_state_dict,
        strict=True
    )

    model.to(device)
    model.eval()

    return model, class_names, device


# =========================================================
# Image Transform
# =========================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# Prediction
# =========================================================

def predict_image(image):

    model, class_names, device = load_model()

    image = image.convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    confidence, predicted_index = torch.max(
        probabilities,
        dim=0
    )

    predicted_class = class_names[
        predicted_index.item()
    ]

    confidence = float(
        confidence.item()
    )

    scores = probabilities.cpu().tolist()

    return (
        predicted_class,
        confidence,
        scores,
        class_names
    )


# =========================================================
# Header
# =========================================================

st.markdown("""
<div class="hero">

<h1>🍌 Banana Leaf AI</h1>

<p>
ระบบ AI สำหรับจำแนกโรคและความเสียหายของใบกล้วย
ด้วยโมเดล MobileNetV2
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.markdown("## 🍌 Banana Leaf AI")

    st.markdown("---")

    page = st.radio(
        "เมนู",
        [
            "🏠 Home",
            "🔍 ทำนาย",
            "📚 ข้อมูลโรค"
        ]
    )

    st.markdown("---")

    st.info(
        "Model: MobileNetV2\n\n"
        "Dataset: Banana Leaf v2"
    )


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.markdown("## 👋 ยินดีต้อนรับ")

    st.write(
        "ระบบสามารถวิเคราะห์ภาพใบกล้วย "
        "และจำแนกประเภทของโรคหรือความเสียหาย "
        "จากภาพที่ผู้ใช้อัปโหลด"
    )

    st.markdown("### 🔍 เริ่มทำนาย")

    st.write(
        "เลือกเมนู **ทำนาย** เพื่ออัปโหลดภาพใบกล้วย"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="info-box">

        ### 📷 1. เลือกรูป

        อัปโหลดภาพใบกล้วย

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="info-box">

        ### 🤖 2. AI วิเคราะห์

        MobileNetV2 วิเคราะห์ภาพ

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="info-box">

        ### 📊 3. ดูผลลัพธ์

        แสดงประเภทและความมั่นใจ

        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PREDICT
# =========================================================

elif page == "🔍 ทำนาย":

    st.markdown("## 🔍 ทำนายโรคใบกล้วย")

    uploaded_file = st.file_uploader(
        "อัปโหลดภาพใบกล้วย",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="ภาพที่เลือก",
                use_container_width=True
            )

        with col2:

            st.markdown(
                "### พร้อมวิเคราะห์หรือยัง?"
            )

            st.write(
                "กดปุ่มด้านล่างเพื่อให้ AI วิเคราะห์"
            )

            if st.button(
                "🔍 ทำนาย",
                use_container_width=True
            ):

                try:

                    with st.spinner(
                        "กำลังวิเคราะห์ภาพ..."
                    ):

                        (
                            predicted_class,
                            confidence,
                            scores,
                            class_names
                        ) = predict_image(image)

                    st.success(
                        "วิเคราะห์เสร็จแล้ว"
                    )

                    st.markdown(
                        '<div class="result-box">',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        "### 🎯 ผลการทำนาย"
                    )

                    st.markdown(
                        f"## {predicted_class}"
                    )

                    st.metric(
                        "ความมั่นใจ",
                        f"{confidence * 100:.2f}%"
                    )

                    # ข้อมูลโรค
                    info = DISEASE_INFO.get(
                        predicted_class
                    )

                    if info:

                        st.markdown(
                            f"**ชื่อภาษาไทย:** {info['thai']}"
                        )

                        st.write(
                            info["description"]
                        )

                        st.markdown(
                            "### 🌱 แนวทางดูแล"
                        )

                        st.write(
                            info["advice"]
                        )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                    # คะแนนทุกคลาส
                    st.markdown(
                        "### 📊 คะแนนของแต่ละประเภท"
                    )

                    for name, score in zip(
                        class_names,
                        scores
                    ):

                        st.write(
                            f"**{name}** — "
                            f"{score * 100:.2f}%"
                        )

                        st.progress(
                            float(score)
                        )

                except FileNotFoundError as e:

                    st.error(
                        str(e)
                    )

                except Exception as e:

                    st.error(
                        "เกิดข้อผิดพลาดในการโหลดโมเดล"
                    )

                    st.code(
                        str(e)
                    )


# =========================================================
# DISEASE INFORMATION
# =========================================================

elif page == "📚 ข้อมูลโรค":

    st.markdown(
        "## 📚 ข้อมูลโรคและความเสียหาย"
    )

    for name, info in DISEASE_INFO.items():

        with st.expander(
            f"{info['thai']} — {name}"
        ):

            st.write(
                f"**รายละเอียด:** {info['description']}"
            )

            st.write(
                f"**แนวทางดูแล:** {info['advice']}"
            )


# =========================================================
# Footer
# =========================================================

st.markdown("---")

st.caption(
    "Banana Leaf AI | MobileNetV2 | Banana Leaf Classification"
)
