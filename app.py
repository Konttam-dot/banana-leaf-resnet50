import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# =========================================================
# Banana Leaf AI - ResNet50
# =========================================================

st.set_page_config(
    page_title="Banana Leaf AI",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# CSS
# -------------------------
st.markdown("""
<style>
    .stApp {
        background: #f7faf8;
    }

    .hero {
        padding: 30px 35px;
        border-radius: 24px;
        background: linear-gradient(135deg, #e8f8ef, #f5fbf7);
        border: 1px solid #d9eee1;
        margin-bottom: 25px;
    }

    .hero h1 {
        margin: 0;
        font-size: 42px;
        color: #183b2a;
    }

    .hero p {
        color: #587064;
        font-size: 17px;
        margin-top: 10px;
    }

    .card {
        background: white;
        border: 1px solid #e3ebe6;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 5px 20px rgba(30, 70, 45, 0.06);
        min-height: 145px;
        margin-bottom: 18px;
    }

    .card h3 {
        color: #183b2a;
        margin-bottom: 8px;
    }

    .card p {
        color: #66756d;
        margin: 0;
    }

    .result-box {
        background: white;
        border-radius: 20px;
        border: 1px solid #e3ebe6;
        padding: 25px;
        margin-top: 15px;
    }

    .healthy {
        background: #eaf8ef;
        border: 1px solid #bfe4ca;
        border-radius: 16px;
        padding: 18px;
    }

    .warning {
        background: #fff7e6;
        border: 1px solid #f3d59b;
        border-radius: 16px;
        padding: 18px;
    }

    .small-note {
        color: #748078;
        font-size: 13px;
    }

    div.stButton > button {
        border-radius: 12px;
        font-weight: 600;
        min-height: 44px;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 16px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# Session State
# สำคัญ: ต้องกำหนดก่อนเรียก st.session_state.page
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# =========================================================
# Classes
# =========================================================
CLASS_NAMES = [
    "Banana Skipper Damage",
    "Black and Yellow Sigatoka",
    "Chewing insect damage on banana leaf",
    "Healthy Banana leaf",
    "Panama Wilt Disease"
]

# ข้อมูลสำหรับหน้า Disease Information
DISEASE_INFO = {
    "Banana Skipper Damage": {
        "icon": "🐛",
        "title": "Banana Skipper Damage",
        "description": "ความเสียหายของใบกล้วยที่เกี่ยวข้องกับหนอนผีเสื้อกล้วย (banana skipper)",
        "advice": "ควรตรวจดูใบและบริเวณต้นกล้วยอย่างสม่ำเสมอ และกำจัดส่วนที่เสียหายตามแนวทางของแหล่งเกษตรในพื้นที่",
        "treatment": "แนวทางดูแล: ตรวจหาและเก็บส่วนของใบที่เสียหายมากออกอย่างเหมาะสม ดูแลความสะอาดแปลง และติดตามการระบาดของแมลง หากพบมากควรปรึกษาเจ้าหน้าที่เกษตรเพื่อเลือกวิธีควบคุมที่เหมาะสม"
    },
    "Black and Yellow Sigatoka": {
        "icon": "🍂",
        "title": "Black and Yellow Sigatoka",
        "description": "โรคใบจุดของกล้วยที่ทำให้เกิดรอยหรือจุดสีเข้มและเหลืองบนใบ",
        "advice": "ควรสำรวจใบที่มีอาการ ตัดใบที่เสียหายตามความเหมาะสม และปรึกษาเจ้าหน้าที่เกษตรเมื่อพบการระบาด",
        "treatment": "แนวทางดูแล: ลดแหล่งสะสมเชื้อด้วยการตัดและจัดการใบที่เป็นโรคอย่างเหมาะสม รักษาระยะปลูกและการระบายอากาศของแปลง และติดตามอาการอย่างสม่ำเสมอ หากการระบาดรุนแรงให้ผู้เชี่ยวชาญแนะนำการควบคุมโรค"
    },
    "Chewing insect damage on banana leaf": {
        "icon": "🐞",
        "title": "Chewing insect Damage",
        "description": "ความเสียหายจากแมลงกัดกิน ทำให้เกิดรอยแหว่งหรือรูบนใบ",
        "advice": "ตรวจดูตัวแมลงและร่องรอยบนใบอย่างสม่ำเสมอ และใช้วิธีควบคุมแมลงที่เหมาะสมกับพื้นที่",
        "treatment": "แนวทางดูแล: สำรวจใต้ใบและบริเวณยอดอย่างสม่ำเสมอ เก็บหรือกำจัดส่วนที่เสียหายมาก และรักษาความสะอาดของแปลง หากพบแมลงจำนวนมากควรปรึกษาเจ้าหน้าที่เกษตรก่อนเลือกวิธีควบคุม"
    },
    "Healthy Banana leaf": {
        "icon": "🌱",
        "title": "Healthy Banana Leaf",
        "description": "ใบกล้วยที่โมเดลจำแนกว่าอยู่ในกลุ่มใบปกติ",
        "advice": "ดูแลน้ำ ปุ๋ย และสภาพแวดล้อมของต้นกล้วยอย่างเหมาะสม พร้อมตรวจใบเป็นประจำ",
        "treatment": "แนวทางดูแล: รักษาความสมบูรณ์ของต้นด้วยน้ำและธาตุอาหารที่เหมาะสม จัดการวัชพืชและเศษใบในแปลง และตรวจใบเป็นประจำเพื่อพบความผิดปกติได้เร็ว"
    },
    "Panama Wilt Disease": {
        "icon": "⚠️",
        "title": "Panama Wilt Disease",
        "description": "โรคเหี่ยวปานามา ซึ่งเป็นโรคสำคัญของกล้วย",
        "advice": "หากพบอาการผิดปกติหลายต้น ควรแยกพื้นที่ที่สงสัยและปรึกษาเจ้าหน้าที่เกษตรหรือผู้เชี่ยวชาญ",
        "treatment": "แนวทางจัดการ: โรคนี้ไม่มีวิธีรักษาต้นที่ติดเชื้อให้กลับมาเป็นปกติได้ง่าย จึงเน้นการป้องกันการแพร่กระจาย เช่น จำกัดการเคลื่อนย้ายดินและอุปกรณ์จากพื้นที่ต้องสงสัย รักษาสุขอนามัยของแปลง และใช้วัสดุปลูกที่ปลอดโรคหรือพันธุ์ที่เหมาะสม โดยควรให้เจ้าหน้าที่เกษตรยืนยันก่อนดำเนินการ"
    }
}


# =========================================================
# Model
# =========================================================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    # รองรับทั้งไฟล์ที่เป็น state_dict และ checkpoint บางรูปแบบ
    model_path = "best_resnet50_banana.pth"

    if not os.path.exists(model_path):
        # รองรับกรณีเก็บไว้ในโฟลเดอร์ models/
        model_path = os.path.join("models", "best_resnet50_banana.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "ไม่พบไฟล์ best_resnet50_banana.pth ใน GitHub Repository "
            "กรุณาอัปโหลดไฟล์โมเดลไว้ที่ root ของ repository"
        )

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    # รองรับ checkpoint ที่มี state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # แก้กรณี key มี "module." จาก DataParallel
    clean_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        clean_state_dict[new_key] = value

    model.load_state_dict(clean_state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, device


# =========================================================
# Transform
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
    model, device = load_model()

    image_rgb = image.convert("RGB")
    tensor = transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    confidence, predicted_index = torch.max(probabilities, dim=0)

    return (
        CLASS_NAMES[predicted_index.item()],
        float(confidence.item()),
        probabilities.cpu().tolist()
    )


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.markdown("## 🍌 Banana Leaf AI")
    st.caption("ระบบจำแนกโรคและความเสียหายของใบกล้วย")

    st.markdown("---")
    st.markdown("### เมนู")

    page = st.radio(
        "ไปยังหน้า",
        ["🏠 Home", "🔍 Predict", "📚 Disease Information"],
        index=["🏠 Home", "🔍 Predict", "📚 Disease Information"].index(
            st.session_state.page
        ),
        label_visibility="collapsed"
    )

    st.session_state.page = page

    st.markdown("---")
    st.markdown("### 🤖 Deep Learning")
    st.info("ResNet50\n\n5 Classes")


# =========================================================
# Header
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🍌 Banana Leaf AI</h1>
    <p>ระบบ AI สำหรับจำแนกโรคและความเสียหายของใบกล้วยด้วย ResNet50</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# HOME
# =========================================================
if st.session_state.page == "🏠 Home":

    st.markdown("## 👋 ยินดีต้อนรับ")
    st.write(
        "อัปโหลดภาพใบกล้วย แล้วให้ AI วิเคราะห์ประเภทของใบกล้วย "
        "พร้อมแสดงเปอร์เซ็นต์ความมั่นใจและข้อมูลประกอบ"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card">
            <h3>📷 วิเคราะห์ภาพ</h3>
            <p>อัปโหลดภาพใบกล้วยจากอุปกรณ์ของคุณ</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 ไปหน้าทำนาย", use_container_width=True):
            st.session_state.page = "🔍 Predict"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="card">
            <h3>🤖 ResNet50</h3>
            <p>โมเดล Deep Learning สำหรับจำแนกใบกล้วย 5 Class</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
            <h3>📊 ผลการวิเคราะห์</h3>
            <p>ดูประเภทและเปอร์เซ็นต์ความมั่นใจของโมเดล</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🌱 ประเภทที่ระบบสามารถจำแนก")

    for i, name in enumerate(CLASS_NAMES, 1):
        info = DISEASE_INFO[name]
        st.write(f"**{i}. {info['icon']} {name}**")


# =========================================================
# PREDICT
# =========================================================
elif st.session_state.page == "🔍 Predict":

    st.markdown("## 🔍 วิเคราะห์ใบกล้วย")
    st.write("อัปโหลดภาพ แล้วกดปุ่ม **วิเคราะห์ภาพ**")

    uploaded = st.file_uploader(
        "📷 เลือกรูปใบกล้วย",
        type=["jpg", "jpeg", "png", "webp"],
        key="banana_uploader"
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.session_state.uploaded_image = image

        left, right = st.columns([1, 1])

        with left:
            st.image(
                image,
                caption="ภาพที่อัปโหลด",
                use_container_width=True
            )

        with right:
            st.markdown("### 🤖 พร้อมวิเคราะห์")
            st.write("ระบบจะใช้ ResNet50 จำแนกภาพเป็น 1 ใน 5 Class")

            if st.button(
                "🔍 วิเคราะห์ภาพ",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("AI กำลังวิเคราะห์ภาพ..."):
                    try:
                        result = predict_image(image)
                        st.session_state.prediction = result
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการโหลดหรือใช้งานโมเดล: {e}")

    # ผลการทำนาย
    if st.session_state.prediction is not None:

        predicted_class, confidence, probabilities = st.session_state.prediction
        info = DISEASE_INFO[predicted_class]

        st.markdown("---")
        st.markdown("## 📊 ผลการวิเคราะห์")

        r1, r2 = st.columns([1, 1])

        with r1:
            st.markdown(
                f"""
                <div class="result-box">
                    <h3>{info['icon']} ผลการทำนาย</h3>
                    <h2>{predicted_class}</h2>
                    <p>ความมั่นใจของโมเดล</p>
                    <h1>{confidence * 100:.2f}%</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        with r2:
            if predicted_class == "Healthy Banana leaf":
                st.markdown(
                    """
                    <div class="healthy">
                        <h3>🌱 ใบกล้วยปกติ</h3>
                        <p>โมเดลจำแนกภาพนี้อยู่ในกลุ่มใบกล้วยปกติ</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="warning">
                        <h3>⚠️ พบความผิดปกติ</h3>
                        <p><b>{predicted_class}</b></p>
                        <p>{info['advice']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("### 📈 ความน่าจะเป็นทั้ง 5 Class")

        chart_data = {
            "Class": CLASS_NAMES,
            "Probability (%)": [p * 100 for p in probabilities]
        }

        st.bar_chart(
            chart_data,
            x="Class",
            y="Probability (%)",
            horizontal=True
        )

        st.markdown("### 📚 ข้อมูล")
        st.info(info["description"])

        st.markdown("### 🩺 แนวทางดูแลเบื้องต้น")
        st.success(info["treatment"])

        st.caption(
            "แนวทางนี้เป็นข้อมูลเบื้องต้นจากหลักการจัดการโรคและศัตรูพืชแบบผสมผสาน ควรให้ผู้เชี่ยวชาญยืนยันโรคก่อนใช้วิธีควบคุมเฉพาะ"
        )

        if st.button(
            "🔄 เริ่มการวิเคราะห์ใหม่",
            use_container_width=True
        ):
            st.session_state.uploaded_image = None
            st.session_state.prediction = None
            st.rerun()

    else:
        st.markdown(
            '<p class="small-note">ยังไม่มีผลการวิเคราะห์ กรุณาอัปโหลดรูปและกดปุ่มวิเคราะห์</p>',
            unsafe_allow_html=True
        )


# =========================================================
# DISEASE INFORMATION
# =========================================================
elif st.session_state.page == "📚 Disease Information":

    st.markdown("## 📚 ข้อมูลโรคและความเสียหาย")
    st.write("เลือกประเภทเพื่อดูข้อมูลเบื้องต้น")

    selected = st.selectbox(
        "เลือกประเภท",
        CLASS_NAMES
    )

    info = DISEASE_INFO[selected]

    st.markdown(
        f"""
        <div class="result-box">
            <h1>{info['icon']} {info['title']}</h1>
            <p>{info['description']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ⚠️ คำแนะนำเบื้องต้น")
    st.warning(info["advice"])

    st.markdown("### 🩺 แนวทางดูแล / จัดการ")
    st.success(info["treatment"])

    st.caption(
        "หมายเหตุ: ผลจาก AI เป็นการจำแนกจากภาพ ไม่ควรใช้แทนการวินิจฉัยโดยผู้เชี่ยวชาญ"
    )
