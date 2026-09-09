import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Banana Leaf AI",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CSS - MODERN GUI
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Prompt', sans-serif;
}

/* Main */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */
.hero {
    padding: 35px;
    border-radius: 25px;
    text-align: center;
    margin-bottom: 30px;
    background: linear-gradient(
        135deg,
        #dff7e8,
        #f3fff7
    );
    border: 1px solid #ccebd8;
}

.hero-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 18px;
}

/* Cards */
.card {
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(128,128,128,0.18);
    background: rgba(255,255,255,0.5);
    margin-bottom: 20px;
}

.card-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 10px;
}

/* Result */
.result-card {
    padding: 30px;
    border-radius: 22px;
    border: 2px solid #b8dfc5;
    text-align: center;
    margin: 20px 0;
}

.result-title {
    font-size: 30px;
    font-weight: 700;
}

.confidence {
    font-size: 35px;
    font-weight: 700;
}

/* Info */
.info-card {
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(128,128,128,0.2);
    margin-top: 20px;
}

/* Upload */
[data-testid="stFileUploader"] {
    border-radius: 15px;
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    min-height: 45px;
}

/* Mobile */
@media (max-width: 768px) {

    .hero {
        padding: 25px 15px;
    }

    .hero-title {
        font-size: 30px;
    }

    .hero-subtitle {
        font-size: 15px;
    }

    .result-title {
        font-size: 23px;
    }

    .confidence {
        font-size: 28px;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# CLASS
# =========================================================

class_names = [
    "Banana Skipper Damage",
    "Black and Yellow Sigatoka",
    "Chewing insect damage on banana leaf",
    "Healthy Banana leaf",
    "Panama Wilt Disease"
]


# =========================================================
# DISEASE DATA
# =========================================================

disease_info = {

    "Banana Skipper Damage": {
        "thai": "ความเสียหายจาก Banana Skipper",
        "cause": "เกิดจากแมลงกลุ่ม Banana Skipper ที่กัดกินใบกล้วย",
        "symptom": "ใบกล้วยมีรอยกัดกินหรือบริเวณเนื้อใบถูกทำลาย",
        "prevention": "ตรวจสอบใบกล้วยเป็นประจำและควบคุมแมลงอย่างเหมาะสม",
        "advice": "ควรตรวจสอบบริเวณใบและต้นกล้วยโดยรอบ"
    },

    "Black and Yellow Sigatoka": {
        "thai": "โรคใบจุดดำและเหลือง",
        "cause": "เกิดจากเชื้อราที่ทำให้เกิดรอยโรคบนใบกล้วย",
        "symptom": "พบจุดหรือรอยแผลสีเข้มและบริเวณสีเหลืองบนใบ",
        "prevention": "ดูแลความสะอาดของแปลงและตัดใบที่เป็นโรค",
        "advice": "หากพบอาการเพิ่มขึ้นควรปรึกษาผู้เชี่ยวชาญ"
    },

    "Chewing insect damage on banana leaf": {
        "thai": "ความเสียหายจากแมลงกัดกินใบ",
        "cause": "เกิดจากแมลงที่กัดกินเนื้อใบกล้วย",
        "symptom": "พบรู รอยแหว่ง หรือบริเวณใบที่ถูกกัดกิน",
        "prevention": "ตรวจสอบแมลงอย่างสม่ำเสมอและควบคุมอย่างเหมาะสม",
        "advice": "ควรตรวจสอบทั้งด้านบนและด้านล่างของใบ"
    },

    "Healthy Banana leaf": {
        "thai": "ใบกล้วยปกติ",
        "cause": "ไม่พบลักษณะความเสียหายตามประเภทที่โมเดลจำแนก",
        "symptom": "ใบมีสภาพสมบูรณ์ ไม่มีรอยโรคหรือรอยกัดกินเด่นชัด",
        "prevention": "ดูแลน้ำ ธาตุอาหาร แสง และตรวจสอบใบเป็นประจำ",
        "advice": "ควรดูแลต้นกล้วยและตรวจสอบใบอย่างต่อเนื่อง"
    },

    "Panama Wilt Disease": {
        "thai": "โรคตายพราย (Panama Wilt)",
        "cause": "เกิดจากเชื้อรา Fusarium ที่สามารถเข้าสู่ระบบท่อลำเลียง",
        "symptom": "ใบอาจเหลือง เหี่ยว และอาการสามารถรุนแรงขึ้น",
        "prevention": "ใช้ต้นพันธุ์ที่สะอาดและระวังการเคลื่อนย้ายดินที่อาจปนเปื้อน",
        "advice": "หากสงสัยว่าเป็นโรคควรปรึกษาผู้เชี่ยวชาญ"
    }
}


# =========================================================
# LOAD MODEL
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


@st.cache_resource
def load_model():

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    model_path = "best_resnet50_banana.pth"

    if not os.path.exists(model_path):

        st.error(
            "❌ ไม่พบไฟล์ best_resnet50_banana.pth"
        )

        st.stop()

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


model = load_model()


# =========================================================
# TRANSFORM
# =========================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================================================
# SESSION STATE
# =========================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🍌 Banana Leaf AI
</div>

<div class="hero-subtitle">
ระบบจำแนกโรคและความเสียหายของใบกล้วยด้วย ResNet50
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    "## 🍌 Banana Leaf AI"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "เมนู",
    [
        "🏠 Home",
        "🔍 Predict",
        "📚 Disease Information"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Deep Learning Model"
)

st.sidebar.success(
    "🤖 ResNet50"
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.header("🏠 ยินดีต้อนรับ")

    st.write(
        "ระบบ AI สำหรับจำแนกประเภทของใบกล้วยจากภาพ "
        "โดยใช้โมเดล ResNet50"
    )

    st.markdown("---")

    # Cards
    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        📷 อัปโหลดภาพ
        </div>

        เลือกรูปใบกล้วยจากอุปกรณ์ของคุณ

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        🤖 AI วิเคราะห์
        </div>

        ResNet50 วิเคราะห์ประเภทของใบกล้วย

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        📊 ดูผลลัพธ์
        </div>

        แสดงประเภทและเปอร์เซ็นต์ความมั่นใจ

        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🌿 ประเภทที่ระบบสามารถจำแนก")

    for i, name in enumerate(class_names, 1):

        st.write(
            f"**{i}.** {name}"
        )

    st.info(
        "💡 ไปที่เมนู 🔍 Predict เพื่อเริ่มวิเคราะห์ใบกล้วย"
    )


# =========================================================
# PREDICT
# =========================================================

elif page == "🔍 Predict":

    st.header("🔍 วิเคราะห์ใบกล้วย")

    st.write(
        "อัปโหลดรูปภาพใบกล้วยเพื่อให้ AI วิเคราะห์"
    )

    uploaded_file = st.file_uploader(
        "📷 เลือกรูปใบกล้วย",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.markdown("---")

        col1, col2 = st.columns(
            [1, 1],
            gap="large"
        )

        # =================================================
        # IMAGE
        # =================================================

        with col1:

            st.subheader("📷 ภาพที่เลือก")

            st.image(
                image,
                use_container_width=True
            )

        # =================================================
        # BUTTON
        # =================================================

        with col2:

            st.subheader("🤖 AI Analysis")

            st.write(
                "พร้อมวิเคราะห์ภาพด้วย ResNet50"
            )

            analyze = st.button(
                "🔍 วิเคราะห์ภาพ",
                type="primary",
                use_container_width=True
            )

            reset = st.button(
                "🔄 เริ่มการวิเคราะห์ใหม่",
                use_container_width=True
            )

        # =================================================
        # RESET
        # =================================================

        if reset:

            st.session_state.prediction = None
            st.session_state.confidence = None
            st.session_state.probabilities = None

            st.rerun()

        # =================================================
        # PREDICT
        # =================================================

        if analyze:

            with st.spinner(
                "🤖 ResNet50 กำลังวิเคราะห์..."
            ):

                tensor = transform(
                    image
                ).unsqueeze(0).to(device)

                with torch.no_grad():

                    output = model(tensor)

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

                confidence_value = (
                    confidence.item() * 100
                )

                st.session_state.prediction = (
                    predicted_class
                )

                st.session_state.confidence = (
                    confidence_value
                )

                st.session_state.probabilities = (
                    probabilities[0]
                    .cpu()
                    .tolist()
                )

        # =================================================
        # RESULT
        # =================================================

        if st.session_state.prediction:

            predicted_class = (
                st.session_state.prediction
            )

            confidence_value = (
                st.session_state.confidence
            )

            probabilities = (
                st.session_state.probabilities
            )

            st.markdown("---")

            st.subheader("🎯 ผลการวิเคราะห์")

            st.markdown(
                f"""
                <div class="result-card">

                <div>
                ผลการทำนาย
                </div>

                <div class="result-title">
                {predicted_class}
                </div>

                <br>

                <div>
                ความมั่นใจของโมเดล
                </div>

                <div class="confidence">
                {confidence_value:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            # =================================================
            # STATUS
            # =================================================

            if predicted_class == "Healthy Banana leaf":

                st.success(
                    "🌱 ใบกล้วยปกติ — "
                    "ไม่พบลักษณะความเสียหายตามประเภทที่โมเดลจำแนก"
                )

            else:

                st.warning(
                    "⚠️ ระบบตรวจพบประเภทที่อาจมีความผิดปกติ"
                )

            # =================================================
            # GRAPH
            # =================================================

            st.markdown("---")

            st.subheader(
                "📈 ความน่าจะเป็นทั้ง 5 Class"
            )

            chart_data = pd.DataFrame({

                "Class": class_names,

                "Probability (%)": [
                    p * 100
                    for p in probabilities
                ]

            })

            chart_data = chart_data.sort_values(
                "Probability (%)",
                ascending=False
            )

            st.bar_chart(
                chart_data.set_index("Class")
            )

            # =================================================
            # PROGRESS
            # =================================================

            st.subheader(
                "📊 รายละเอียดคะแนน"
            )

            for i, name in enumerate(class_names):

                percent = (
                    probabilities[i] * 100
                )

                st.write(
                    f"**{name}** — {percent:.2f}%"
                )

                st.progress(
                    min(int(percent), 100)
                )

            # =================================================
            # DISEASE INFO
            # =================================================

            info = disease_info[
                predicted_class
            ]

            st.markdown("---")

            st.subheader(
                "📚 ข้อมูลที่เกี่ยวข้อง"
            )

            st.markdown(
                f"""
                <div class="info-card">

                <h3>{info['thai']}</h3>

                <b>🔬 สาเหตุ</b><br>
                {info['cause']}<br><br>

                <b>👀 ลักษณะ</b><br>
                {info['symptom']}<br><br>

                <b>🛡️ การป้องกัน</b><br>
                {info['prevention']}<br><br>

                <b>💡 คำแนะนำ</b><br>
                {info['advice']}

                </div>
                """,
                unsafe_allow_html=True
            )

            if predicted_class != "Healthy Banana leaf":

                st.warning(
                    "⚠️ ผลนี้เป็นการประเมินจากโมเดล AI "
                    "ควรใช้ผู้เชี่ยวชาญยืนยันก่อนดำเนินการจริง"
                )


# =========================================================
# DISEASE INFORMATION
# =========================================================

else:

    st.header("📚 Disease Information")

    st.write(
        "เลือกประเภทเพื่อดูข้อมูลเกี่ยวกับโรคหรือความเสียหาย"
    )

    selected = st.selectbox(
        "🌿 เลือกประเภท",
        class_names
    )

    info = disease_info[selected]

    st.markdown("---")

    st.markdown(
        f"""
        <div class="info-card">

        <h2>🍌 {info['thai']}</h2>

        <br>

        <h4>🔬 สาเหตุ</h4>
        {info['cause']}

        <br><br>

        <h4>👀 ลักษณะ</h4>
        {info['symptom']}

        <br><br>

        <h4>🛡️ การป้องกัน</h4>
        {info['prevention']}

        <br><br>

        <h4>💡 คำแนะนำ</h4>
        {info['advice']}

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    if selected == "Healthy Banana leaf":

        st.success(
            "🌱 ใบกล้วยปกติ"
        )

    else:

        st.warning(
            "⚠️ ข้อมูลนี้ใช้เพื่อประกอบการเรียนรู้ "
            "และการประเมินเบื้องต้น"
        )
