import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Banana Leaf AI",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #f7faf8;
    }

    /* Hide default Streamlit elements */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0f3d2e 0%,
            #145c43 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Hero */
    .hero {
        padding: 45px 35px;
        border-radius: 25px;
        background: linear-gradient(
            135deg,
            #dff8e9,
            #f5fff9
        );
        border: 1px solid #c9ead7;
        margin-bottom: 30px;
    }

    .hero-title {
        font-size: 48px;
        font-weight: 800;
        color: #12372a;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 19px;
        color: #4b6359;
    }

    /* Cards */
    .card {
        background: white;
        padding: 28px;
        border-radius: 20px;
        border: 1px solid #e3ebe6;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    .card:hover {
        box-shadow: 0 8px 28px rgba(0,0,0,0.08);
    }

    .card-title {
        font-size: 22px;
        font-weight: 700;
        color: #173d30;
    }

    .card-text {
        color: #65756d;
        font-size: 16px;
    }

    /* Result */
    .result-normal {
        background: #e6f8ed;
        border-left: 7px solid #28a866;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
    }

    .result-disease {
        background: #fff4e5;
        border-left: 7px solid #f39c12;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
    }

    .result-title {
        font-size: 27px;
        font-weight: 800;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        min-height: 45px;
        font-weight: 700;
    }

    /* Mobile */
    @media (max-width: 768px) {

        .hero-title {
            font-size: 34px;
        }

        .hero {
            padding: 28px 20px;
        }

        .card {
            padding: 20px;
        }

    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "Banana Skipper Damage",
    "Black and Yellow Sigatoka",
    "Chewing insect damage on banana leaf",
    "Healthy Banana leaf",
    "Panama Wilt Disease"
]


# =========================================================
# DISEASE INFORMATION
# =========================================================

disease_info = {

    "Banana Skipper Damage": {
        "icon": "🐛",
        "name": "Banana Skipper Damage",
        "description": "ความเสียหายที่เกิดจากหนอนหรือแมลงศัตรูใบกล้วย",
        "advice": "ควรตรวจสอบบริเวณใบและกำจัดส่วนที่เสียหาย พร้อมเฝ้าระวังการแพร่กระจายของแมลง"
    },

    "Black and Yellow Sigatoka": {
        "icon": "🦠",
        "name": "Black and Yellow Sigatoka",
        "description": "โรคใบจุดที่ทำให้เกิดรอยจุดหรือแถบสีเข้มบนใบกล้วย",
        "advice": "ควรตัดใบที่มีอาการรุนแรงออก ลดความชื้นในแปลง และติดตามอาการของใบใหม่"
    },

    "Chewing insect damage on banana leaf": {
        "icon": "🐛",
        "name": "Chewing insect damage",
        "description": "ความเสียหายจากแมลงที่กัดกินเนื้อใบ ทำให้เกิดรูหรือรอยแหว่ง",
        "advice": "ควรตรวจสอบใต้ใบและบริเวณรอบต้นเพื่อค้นหาแมลง และจัดการตามวิธีที่เหมาะสม"
    },

    "Healthy Banana leaf": {
        "icon": "🌱",
        "name": "Healthy Banana Leaf",
        "description": "ใบกล้วยที่ไม่พบลักษณะความเสียหายหรือโรคตามที่โมเดลจำแนก",
        "advice": "ใบกล้วยอยู่ในกลุ่มปกติ ควรดูแลน้ำ แสง และธาตุอาหารให้เหมาะสม"
    },

    "Panama Wilt Disease": {
        "icon": "⚠️",
        "name": "Panama Wilt Disease",
        "description": "โรคเหี่ยวที่เกี่ยวข้องกับเชื้อราที่ส่งผลต่อระบบท่อลำเลียงของต้นกล้วย",
        "advice": "ควรแยกต้นที่สงสัยออกจากบริเวณปลูกและปรึกษาผู้เชี่ยวชาญด้านพืชเพื่อยืนยันการวินิจฉัย"
    }
}


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = "best_resnet50_banana.pth"


# =========================================================
# TRANSFORM
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
# LOAD RESNET50
# =========================================================

@st.cache_resource
def load_model():

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    if not os.path.exists(MODEL_PATH):
        return None

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # กรณี state_dict โดยตรง
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(
        checkpoint,
        strict=True
    )

    model = model.to(device)
    model.eval()

    return model


model = load_model()


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="
        text-align:center;
        padding:20px 0;
    ">
        <div style="font-size:55px;">🍌</div>
        <h2>Banana Leaf AI</h2>
        <p style="color:#d5eee1 !important;">
        AI Classification System
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📌 เมนู")

    page = st.radio(
        "",
        [
            "🏠 Home",
            "🔍 Predict",
            "📚 Disease Information"
        ],
        key="page"
    )

    st.divider()

    st.caption(
        f"🤖 Model: ResNet50\n\n"
        f"💻 Device: {device}"
    )


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">

        <div class="hero-title">
            🍌 Banana Leaf AI
        </div>

        <div class="hero-subtitle">
            ระบบ AI สำหรับจำแนกโรคและความเสียหายของใบกล้วย
            ด้วยโมเดล ResNet50
        </div>

    </div>
    """, unsafe_allow_html=True)


    st.markdown("## 👋 ยินดีต้อนรับ")

    st.write(
        "อัปโหลดภาพใบกล้วยเพื่อให้ระบบ AI วิเคราะห์ประเภทของใบกล้วย "
        "พร้อมแสดงเปอร์เซ็นต์ความมั่นใจและข้อมูลเกี่ยวกับโรค"
    )

    st.write("")


    # Feature cards
    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        📷 วิเคราะห์ภาพ
        </div>

        <div class="card-text">
        อัปโหลดรูปใบกล้วยและให้ AI วิเคราะห์
        </div>

        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "🔍 เริ่มทำนาย",
            use_container_width=True,
            type="primary"
        ):

            st.session_state.page = "🔍 Predict"
            st.rerun()


    with col2:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        📊 ดูผลวิเคราะห์
        </div>

        <div class="card-text">
        ดูประเภทใบกล้วยและระดับความมั่นใจ
        </div>

        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "📊 ไปหน้าวิเคราะห์",
            use_container_width=True
        ):

            st.session_state.page = "🔍 Predict"
            st.rerun()


    with col3:

        st.markdown("""
        <div class="card">

        <div class="card-title">
        📚 ข้อมูลโรค
        </div>

        <div class="card-text">
        ศึกษาข้อมูลโรคและความเสียหายของใบกล้วย
        </div>

        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "📚 ดูข้อมูลโรค",
            use_container_width=True
        ):

            st.session_state.page = "📚 Disease Information"
            st.rerun()


    st.divider()

    st.markdown("## 🌱 ประเภทที่ระบบสามารถจำแนก")

    cols = st.columns(5)

    for i, name in enumerate(class_names):

        with cols[i]:

            info = disease_info[name]

            st.markdown(
                f"""
                <div class="card"
                     style="text-align:center;">

                    <div style="font-size:35px;">
                        {info["icon"]}
                    </div>

                    <b>{info["name"]}</b>

                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# PREDICT
# =========================================================

elif page == "🔍 Predict":

    st.markdown("""
    <div class="hero">

        <div class="hero-title">
            🔍 วิเคราะห์ใบกล้วย
        </div>

        <div class="hero-subtitle">
            อัปโหลดภาพ แล้วให้ ResNet50 ช่วยจำแนกประเภท
        </div>

    </div>
    """, unsafe_allow_html=True)


    if model is None:

        st.error(
            "❌ ไม่พบไฟล์โมเดล "
            f"`{MODEL_PATH}`"
        )

        st.info(
            "ให้นำไฟล์ best_resnet50_banana.pth "
            "ไว้ในโฟลเดอร์เดียวกับ app.py"
        )

    else:

        uploaded_file = st.file_uploader(
            "📷 เลือกรูปใบกล้วย",
            type=["jpg", "jpeg", "png"],
            help="รองรับไฟล์ JPG, JPEG และ PNG"
        )


        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")


            col1, col2 = st.columns(
                [1, 1]
            )


            with col1:

                st.markdown(
                    "### 📷 ภาพที่อัปโหลด"
                )

                st.image(
                    image,
                    use_container_width=True
                )


            with col2:

                st.markdown(
                    "### 🤖 พร้อมวิเคราะห์"
                )

                st.write(
                    "ระบบจะใช้ ResNet50 "
                    "วิเคราะห์ภาพใบกล้วย"
                )

                analyze = st.button(
                    "🔍 วิเคราะห์ภาพ",
                    use_container_width=True,
                    type="primary"
                )


                if analyze:

                    with st.spinner(
                        "🤖 AI กำลังวิเคราะห์..."
                    ):

                        input_tensor = transform(
                            image
                        ).unsqueeze(0)

                        input_tensor = input_tensor.to(
                            device
                        )


                        with torch.no_grad():

                            output = model(
                                input_tensor
                            )

                            probabilities = torch.softmax(
                                output,
                                dim=1
                            )[0]


                        confidence, predicted = torch.max(
                            probabilities,
                            0
                        )


                        predicted_class = class_names[
                            predicted.item()
                        ]

                        confidence_value = (
                            confidence.item() * 100
                        )


                        st.session_state.prediction_done = True

                        st.session_state.prediction = predicted_class

                        st.session_state.confidence = confidence_value

                        st.session_state.probabilities = (
                            probabilities.cpu().numpy()
                        )


                        st.rerun()


    # =====================================================
    # RESULT
    # =====================================================

    if st.session_state.prediction_done:

        predicted_class = st.session_state.prediction

        confidence_value = st.session_state.confidence

        probabilities = st.session_state.probabilities


        st.divider()

        st.markdown("## 📊 ผลการวิเคราะห์")


        info = disease_info[
            predicted_class
        ]


        # Normal / Disease
        if predicted_class == "Healthy Banana leaf":

            st.markdown(
                f"""
                <div class="result-normal">

                    <div class="result-title">
                    🌱 {info["name"]}
                    </div>

                    <p>
                    ระบบจัดอยู่ในกลุ่มใบกล้วยปกติ
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-disease">

                    <div class="result-title">
                    ⚠️ {info["name"]}
                    </div>

                    <p>
                    ระบบตรวจพบลักษณะที่อยู่ในกลุ่ม
                    ความเสียหายหรือโรค
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "🎯 ความมั่นใจ",
                f"{confidence_value:.2f}%"
            )


        with col2:

            st.metric(
                "🏷️ ประเภท",
                info["name"]
            )


        # =================================================
        # PROBABILITY GRAPH
        # =================================================

        st.markdown(
            "### 📈 ความน่าจะเป็นของแต่ละ Class"
        )


        percentages = probabilities * 100


        df = pd.DataFrame({
            "Class": class_names,
            "Probability": percentages
        })


        df = df.sort_values(
            "Probability",
            ascending=True
        )


        fig, ax = plt.subplots(
            figsize=(10, 5)
        )


        ax.barh(
            df["Class"],
            df["Probability"]
        )

        ax.set_xlabel(
            "Probability (%)"
        )

        ax.set_xlim(
            0,
            100
        )

        ax.grid(
            axis="x",
            alpha=0.2
        )


        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


        # =================================================
        # ADVICE
        # =================================================

        st.markdown(
            "### 💡 คำแนะนำ"
        )


        if predicted_class == "Healthy Banana leaf":

            st.success(
                f"🌱 {info['advice']}"
            )

        else:

            st.warning(
                f"⚠️ {info['advice']}"
            )


        # =================================================
        # NEW ANALYSIS
        # =================================================

        st.write("")


        if st.button(
            "🔄 เริ่มการวิเคราะห์ใหม่",
            use_container_width=True
        ):

            st.session_state.prediction_done = False

            st.session_state.prediction = None

            st.session_state.confidence = None

            st.session_state.probabilities = None

            st.rerun()


# =========================================================
# DISEASE INFORMATION
# =========================================================

elif page == "📚 Disease Information":

    st.markdown("""
    <div class="hero">

        <div class="hero-title">
            📚 Disease Information
        </div>

        <div class="hero-subtitle">
            ข้อมูลประเภทโรคและความเสียหายของใบกล้วย
        </div>

    </div>
    """, unsafe_allow_html=True)


    for name in class_names:

        info = disease_info[name]

        with st.expander(
            f"{info['icon']} {info['name']}"
        ):

            st.markdown(
                f"### {info['icon']} {info['name']}"
            )

            st.write(
                info["description"]
            )

            st.markdown(
                "**💡 คำแนะนำ**"
            )

            st.info(
                info["advice"]
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<br>
<hr>

<div style="
    text-align:center;
    color:#7a8981;
    font-size:14px;
">

🍌 <b>Banana Leaf AI</b><br>
Banana Leaf Classification using ResNet50<br>
AI-assisted plant disease classification system

</div>
""", unsafe_allow_html=True)
