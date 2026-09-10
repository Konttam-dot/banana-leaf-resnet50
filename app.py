import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


# =========================================================
# ตั้งค่าหน้าเว็บ
# =========================================================

st.set_page_config(
    page_title="Banana Leaf AI",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CSS - ทำให้หน้าตาทันสมัย
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7faf8;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Header */
.hero {
    background: linear-gradient(135deg, #e8f8ef, #f8fffb);
    padding: 35px;
    border-radius: 25px;
    border: 1px solid #d5eee0;
    text-align: center;
    margin-bottom: 30px;
}

.hero-title {
    font-size: 45px;
    font-weight: 800;
    color: #183b2a;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 18px;
    color: #587064;
}

/* Card */
.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #e2e9e5;
    box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.card h3 {
    color: #183b2a;
}

/* Result */
.result-normal {
    background: #e9f8ef;
    border: 2px solid #9edbb5;
    padding: 25px;
    border-radius: 20px;
    margin-top: 20px;
}

.result-disease {
    background: #fff5f3;
    border: 2px solid #f0aaa0;
    padding: 25px;
    border-radius: 20px;
    margin-top: 20px;
}

.big-result {
    font-size: 30px;
    font-weight: 800;
}

.confidence {
    font-size: 42px;
    font-weight: 800;
    color: #183b2a;
}

/* Mobile */
@media (max-width: 768px) {

    .hero-title {
        font-size: 32px;
    }

    .hero-subtitle {
        font-size: 15px;
    }

    .big-result {
        font-size: 24px;
    }

    .confidence {
        font-size: 34px;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# Class ของ Dataset
# สำคัญ: ต้องตรงกับตอน Train
# =========================================================

CLASS_NAMES = [
    "Banana Skipper Damage",
    "Black and Yellow Sigatoka",
    "Chewing insect damage on banana leaf",
    "Healthy Banana leaf",
    "Panama Wilt Disease"
]


# =========================================================
# ข้อมูลโรค
# =========================================================

DISEASE_INFO = {

    "Banana Skipper Damage": {
        "icon": "🦋",
        "title": "Banana Skipper Damage",
        "description": """
ความเสียหายที่เกี่ยวข้องกับแมลง Banana Skipper
ซึ่งอาจทำให้ใบกล้วยมีร่องรอยหรือส่วนของใบได้รับความเสียหาย
""",
        "advice": """
🌿 ควรตรวจสอบบริเวณใบและต้นกล้วยอย่างสม่ำเสมอ
และกำจัดส่วนของใบที่เสียหายมากตามความเหมาะสม
"""
    },

    "Black and Yellow Sigatoka": {
        "icon": "🦠",
        "title": "Black and Yellow Sigatoka",
        "description": """
โรคใบกล้วยที่มีลักษณะเป็นจุดหรือรอยผิดปกติบนใบ
และสามารถทำให้พื้นที่สีเขียวของใบลดลง
""",
        "advice": """
🌿 ควรตัดแต่งใบที่เป็นโรคและลดความชื้นบริเวณแปลง
พร้อมติดตามอาการของใบกล้วยอย่างต่อเนื่อง
"""
    },

    "Chewing insect damage on banana leaf": {
        "icon": "🐛",
        "title": "Chewing insect damage",
        "description": """
ความเสียหายจากแมลงกัดกินใบ
มักสังเกตได้จากบริเวณใบที่มีร่องรอยการกัดกิน
""",
        "advice": """
🌿 ตรวจสอบตัวแมลงหรือร่องรอยบนใบ
และจัดการแมลงตามวิธีที่เหมาะสมกับการปลูกกล้วย
"""
    },

    "Healthy Banana leaf": {
        "icon": "🌱",
        "title": "Healthy Banana leaf",
        "description": """
ใบกล้วยปกติ ไม่พบลักษณะที่ตรงกับ Class
โรคหรือความเสียหายที่โมเดลได้รับการฝึกให้จำแนก
""",
        "advice": """
✅ ใบกล้วยถูกจำแนกเป็นใบปกติ
ควรดูแลต้นกล้วยตามปกติและตรวจสอบใบอย่างสม่ำเสมอ
"""
    },

    "Panama Wilt Disease": {
        "icon": "🦠",
        "title": "Panama Wilt Disease",
        "description": """
โรคเหี่ยวของกล้วยที่เกี่ยวข้องกับเชื้อราในดิน
สามารถส่งผลต่อระบบท่อลำเลียงน้ำของต้นกล้วย
""",
        "advice": """
⚠️ หากพบอาการผิดปกติควรแยกและตรวจสอบต้นที่สงสัย
และปรึกษาผู้เชี่ยวชาญด้านโรคพืชเพื่อยืนยันสาเหตุ
"""
    }
}


# =========================================================
# Path ของ Model
# =========================================================

MODEL_PATHS = [
    "best_resnet50_banana.pth",
    "best_resnet50.pth",
    "model/best_resnet50_banana.pth",
    "models/best_resnet50_banana.pth"
]


def find_model():

    for path in MODEL_PATHS:

        if os.path.exists(path):
            return path

    return None


MODEL_PATH = find_model()


# =========================================================
# Device
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# Transform
# ต้องใกล้เคียงกับตอน Train
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
# สร้าง ResNet50
# =========================================================

@st.cache_resource
def create_model():

    model = models.resnet50(weights=None)

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        len(CLASS_NAMES)
    )

    if MODEL_PATH is None:
        return model, False, None

    try:

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

        # กรณี checkpoint เป็น state_dict โดยตรง
        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            else:
                state_dict = checkpoint

        else:
            state_dict = checkpoint

        # รองรับกรณีชื่อ key มี module.
        new_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("module."):
                key = key.replace("module.", "", 1)

            new_state_dict[key] = value

        model.load_state_dict(
            new_state_dict,
            strict=True
        )

        model.to(DEVICE)
        model.eval()

        return model, True, None

    except Exception as e:

        return model, False, str(e)


model, model_loaded, model_error = create_model()


# =========================================================
# Function Predict
# =========================================================

def predict_image(image):

    image_rgb = image.convert("RGB")

    image_tensor = transform(image_rgb)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = probabilities[
        predicted_index
    ].item()

    return (
        predicted_class,
        confidence,
        probabilities.cpu()
    )


# =========================================================
# Header
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🍌 Banana Leaf AI
</div>

<div class="hero-subtitle">
ระบบ AI สำหรับจำแนกโรคและความเสียหายของใบกล้วยด้วย ResNet50
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# Sidebar
# =========================================================

st.sidebar.markdown(
    "# 🍌 Banana Leaf AI"
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

st.sidebar.info(
    f"""
🤖 Model: ResNet50

📊 Classes: 5

💻 Device: {DEVICE}
"""
)


# =========================================================
# ตรวจสอบ Model
# =========================================================

if not model_loaded:

    if MODEL_PATH is None:

        st.warning(
            "⚠️ ไม่พบไฟล์โมเดล ResNet50 ใน GitHub"
        )

        st.info(
            "กรุณาอัปโหลดไฟล์ "
            "`best_resnet50_banana.pth` "
            "ไว้ใน Repository เดียวกับ app.py"
        )

    else:

        st.error(
            "❌ ไม่สามารถโหลดโมเดลได้"
        )

        if model_error:
            st.code(model_error)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.markdown(
        "## 👋 ยินดีต้อนรับ"
    )

    st.write(
        "ระบบสามารถวิเคราะห์ภาพใบกล้วยและจำแนกออกเป็น 5 Class "
        "พร้อมแสดงเปอร์เซ็นต์ความมั่นใจของโมเดล"
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="card">

        <h3>📷 วิเคราะห์ภาพ</h3>

        <p>
        อัปโหลดรูปใบกล้วยเพื่อให้ AI วิเคราะห์
        </p>

        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "🔍 เริ่มทำนาย",
            use_container_width=True
        ):

            st.session_state["page"] = "predict"

            st.switch_page(
                st.query_params.get(
                    "page",
                    "app.py"
                )
            )

    with col2:

        st.markdown("""
        <div class="card">

        <h3>🤖 ResNet50</h3>

        <p>
        โมเดล Deep Learning สำหรับจำแนกใบกล้วย
        </p>

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="card">

        <h3>📊 ผลการวิเคราะห์</h3>

        <p>
        แสดงผลการทำนายและความมั่นใจ
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
        "## 🌱 ประเภทที่ระบบสามารถจำแนก"
    )

    for i, class_name in enumerate(CLASS_NAMES):

        info = DISEASE_INFO[class_name]

        if class_name == "Healthy Banana leaf":

            st.success(
                f"🌱 **{class_name}** — ใบกล้วยปกติ"
            )

        else:

            st.write(
                f"{info['icon']} **{class_name}**"
            )


# =========================================================
# PREDICT
# =========================================================

elif page == "🔍 Predict":

    st.markdown(
        "## 🔍 วิเคราะห์ใบกล้วย"
    )

    st.write(
        "อัปโหลดภาพใบกล้วย จากนั้นกดปุ่ม **วิเคราะห์ภาพ**"
    )

    uploaded_file = st.file_uploader(
        "📷 เลือกรูปใบกล้วย",
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
        )

        st.image(
            image,
            caption="ภาพที่เลือก",
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:

            analyze = st.button(
                "🔍 วิเคราะห์ภาพ",
                use_container_width=True,
                type="primary"
            )

        with col2:

            clear = st.button(
                "🔄 เริ่มการวิเคราะห์ใหม่",
                use_container_width=True
            )

        if clear:

            st.rerun()

        if analyze:

            if not model_loaded:

                st.error(
                    "❌ ยังไม่สามารถใช้งานโมเดลได้"
                )

            else:

                with st.spinner(
                    "🤖 ResNet50 กำลังวิเคราะห์..."
                ):

                    predicted_class, confidence, probabilities = predict_image(
                        image
                    )

                confidence_percent = confidence * 100

                # ==========================================
                # HEALTHY
                # ==========================================

                if predicted_class == "Healthy Banana leaf":

                    st.markdown(
                        f"""
                        <div class="result-normal">

                        <div class="big-result">
                        🌱 ใบกล้วยปกติ
                        </div>

                        <br>

                        <b>ผลการทำนาย</b>

                        <br>

                        Healthy Banana leaf

                        <br><br>

                        <b>ความมั่นใจของโมเดล</b>

                        <div class="confidence">
                        {confidence_percent:.2f}%
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.success(
                        "✅ ระบบจำแนกภาพนี้เป็นใบกล้วยปกติ"
                    )

                    st.info(
                        "🌱 ไม่พบลักษณะที่ตรงกับ Class "
                        "โรคหรือความเสียหายที่โมเดลได้รับการฝึกให้จำแนก"
                    )

                # ==========================================
                # DISEASE / DAMAGE
                # ==========================================

                else:

                    info = DISEASE_INFO[
                        predicted_class
                    ]

                    st.markdown(
                        f"""
                        <div class="result-disease">

                        <div class="big-result">
                        {info['icon']} พบความผิดปกติ
                        </div>

                        <br>

                        <b>ผลการทำนาย</b>

                        <br>

                        {predicted_class}

                        <br><br>

                        <b>ความมั่นใจของโมเดล</b>

                        <div class="confidence">
                        {confidence_percent:.2f}%
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.warning(
                        "⚠️ ภาพนี้ถูกจำแนกเป็นกลุ่มที่มีความผิดปกติ"
                    )

                    st.markdown(
                        "### ⚠️ คำแนะนำ"
                    )

                    st.write(
                        info["advice"]
                    )

                # ==========================================
                # Probability
                # ==========================================

                st.markdown("---")

                st.markdown(
                    "### 📈 ความน่าจะเป็นของทั้ง 5 Class"
                )

                probability_dict = {}

                for i, class_name in enumerate(CLASS_NAMES):

                    probability_dict[
                        class_name
                    ] = float(
                        probabilities[i].item()
                    )

                st.bar_chart(
                    probability_dict
                )

                # ==========================================
                # ตาราง
                # ==========================================

                st.markdown(
                    "### 📊 รายละเอียดความมั่นใจ"
                )

                for i, class_name in enumerate(CLASS_NAMES):

                    percent = (
                        probabilities[i].item()
                        * 100
                    )

                    if class_name == "Healthy Banana leaf":

                        st.write(
                            f"🌱 {class_name}: "
                            f"**{percent:.2f}%**"
                        )

                    else:

                        st.write(
                            f"🔹 {class_name}: "
                            f"**{percent:.2f}%**"
                        )


# =========================================================
# DISEASE INFORMATION
# =========================================================

elif page == "📚 Disease Information":

    st.markdown(
        "## 📚 ข้อมูลโรคและความเสียหายของใบกล้วย"
    )

    st.write(
        "เลือกประเภทที่ต้องการดูข้อมูล"
    )

    selected_class = st.selectbox(
        "เลือกประเภท",
        CLASS_NAMES
    )

    info = DISEASE_INFO[
        selected_class
    ]

    st.markdown("---")

    if selected_class == "Healthy Banana leaf":

        st.success(
            f"🌱 {info['title']}"
        )

    else:

        st.warning(
            f"{info['icon']} {info['title']}"
        )

    st.markdown(
        f"""
        <div class="card">

        <h3>
        {info['icon']} {info['title']}
        </h3>

        <p>
        {info['description']}
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### 💡 คำแนะนำ"
    )

    st.info(
        info["advice"]
    )

    st.markdown("---")

    st.markdown(
        "### 📌 หมายเหตุ"
    )

    st.caption(
        "ผลการทำนายเป็นผลจากโมเดล ResNet50 "
        "และควรใช้เป็นข้อมูลประกอบการตรวจสอบเบื้องต้น "
        "ไม่ควรใช้แทนการวินิจฉัยโดยผู้เชี่ยวชาญด้านโรคพืช"
    )
