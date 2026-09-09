import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd


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

.stApp {
    background-color: #f5f8f6;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #123c2c;
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* Hero */
.hero {
    background: linear-gradient(
        135deg,
        #dff7e8,
        #f8fffb
    );

    padding: 35px;
    border-radius: 25px;

    text-align: center;

    border: 1px solid #ccebd8;

    margin-bottom: 30px;
}

.hero h1 {
    font-size: 42px;
    color: #173b2d;
    margin-bottom: 10px;
}

.hero p {
    font-size: 18px;
    color: #557065;
}


/* Card */

.card {
    background-color: white;

    padding: 25px;

    border-radius: 20px;

    border: 1px solid #e2e9e5;

    box-shadow:
        0px 5px 20px rgba(0,0,0,0.05);

    min-height: 150px;

    margin-bottom: 15px;
}

.card h3 {
    color: #173b2d;
}

.card p {
    color: #687870;
}


/* Result */

.result-box {
    background-color: white;

    padding: 30px;

    border-radius: 20px;

    border: 1px solid #dfe8e2;

    box-shadow:
        0px 5px 20px rgba(0,0,0,0.05);

    margin-top: 20px;
}


/* Buttons */

.stButton > button {
    border-radius: 12px;

    height: 45px;

    font-weight: 600;
}


/* Upload */

[data-testid="stFileUploader"] {
    background-color: white;

    padding: 15px;

    border-radius: 15px;

    border: 1px solid #e2e9e5;
}


/* Metric */

[data-testid="stMetric"] {
    background-color: white;

    padding: 15px;

    border-radius: 15px;
}


/* Footer */

.footer {
    text-align: center;

    color: #7b8982;

    padding: 30px;

    margin-top: 40px;
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

    "Banana Skipper Damage":
        """
        ความเสียหายของใบกล้วยที่เกิดจากแมลง
        อาจพบรอยกัดหรือส่วนของใบที่ถูกทำลาย
        """,

    "Black and Yellow Sigatoka":
        """
        โรคที่เกี่ยวข้องกับเชื้อราบนใบกล้วย
        มักทำให้เกิดจุดหรือรอยสีเข้มบนใบ
        """,

    "Chewing insect damage on banana leaf":
        """
        ความเสียหายจากแมลงที่กัดกินใบ
        ทำให้เกิดรูหรือรอยแหว่งบนใบกล้วย
        """,

    "Healthy Banana leaf":
        """
        ใบกล้วยที่อยู่ในสภาพปกติ
        ไม่พบลักษณะความเสียหายตามกลุ่มที่โมเดลจำแนก
        """,

    "Panama Wilt Disease":
        """
        โรคเหี่ยวของกล้วยที่เกิดจากเชื้อราในดิน
        สามารถส่งผลต่อระบบท่อลำเลียงของต้นกล้วย
        """
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
# IMAGE TRANSFORM
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
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    # สร้าง ResNet50
    model = models.resnet50(
        weights=None
    )

    # เปลี่ยน Layer สุดท้าย
    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    # ตรวจสอบไฟล์โมเดล
    if not os.path.exists(MODEL_PATH):

        return None

    # โหลดโมเดล
    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    # ส่งโมเดลไป CPU/GPU
    model.to(device)

    # Evaluation mode
    model.eval()

    return model


model = load_model()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center">

        <div style="font-size:55px">
        🍌
        </div>

        <h2>Banana Leaf AI</h2>

        <p>
        AI Classification
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "เมนู",
        [
            "🏠 Home",
            "🔍 Predict",
            "📚 Disease Information"
        ]
    )

    st.divider()

    st.markdown(
        f"""
        **🤖 Model**

        ResNet50

        **💻 Device**

        {device}
        """
    )


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.markdown(
        """
        <div class="hero">

        <h1>
        🍌 Banana Leaf AI
        </h1>

        <p>
        ระบบ AI สำหรับจำแนกโรคและความเสียหาย
        ของใบกล้วยด้วย ResNet50
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "## 👋 ยินดีต้อนรับ"
    )

    st.write(
        """
        ระบบนี้สามารถวิเคราะห์ภาพใบกล้วย
        และจำแนกประเภทออกเป็น 5 Class
        พร้อมแสดงเปอร์เซ็นต์ความมั่นใจ
        """
    )


    st.write("")


    # Cards

    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            """
            <div class="card">

            <h3>
            📷 วิเคราะห์ภาพ
            </h3>

            <p>
            อัปโหลดรูปใบกล้วย
            เพื่อให้ AI วิเคราะห์
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="card">

            <h3>
            🤖 ResNet50
            </h3>

            <p>
            โมเดล Deep Learning
            สำหรับจำแนกใบกล้วย
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="card">

            <h3>
            📊 ผลการวิเคราะห์
            </h3>

            <p>
            แสดงผลการทำนาย
            และเปอร์เซ็นต์ความมั่นใจ
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.divider()


    st.markdown(
        "## 🌱 ประเภทที่ระบบสามารถจำแนก"
    )


    for name in class_names:

        st.markdown(
            f"""
            <div class="card">

            <h3>
            🌿 {name}
            </h3>

            <p>
            {disease_info[name]}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# PREDICT
# =========================================================

elif page == "🔍 Predict":

    st.markdown(
        """
        <div class="hero">

        <h1>
        🔍 วิเคราะห์ใบกล้วย
        </h1>

        <p>
        อัปโหลดภาพใบกล้วย
        แล้วให้ ResNet50 วิเคราะห์
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ตรวจสอบโมเดล

    if model is None:

        st.error(
            "❌ ไม่พบไฟล์ best_resnet50_banana.pth"
        )

        st.warning(
            """
            กรุณาตรวจสอบว่าไฟล์

            best_resnet50_banana.pth

            อยู่ใน GitHub โฟลเดอร์เดียวกับ app.py
            """
        )


    else:

        # Upload

        uploaded_file = st.file_uploader(
            "📷 เลือกรูปใบกล้วย",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )


        if uploaded_file is not None:

            # เปิดรูป

            image = Image.open(
                uploaded_file
            ).convert("RGB")


            col1, col2 = st.columns(
                [1, 1]
            )


            # ==========================================
            # IMAGE
            # ==========================================

            with col1:

                st.markdown(
                    "### 📷 ภาพที่อัปโหลด"
                )

                st.image(
                    image,
                    use_container_width=True
                )


            # ==========================================
            # ANALYSIS
            # ==========================================

            with col2:

                st.markdown(
                    "### 🤖 AI Analysis"
                )

                st.write(
                    """
                    ระบบพร้อมวิเคราะห์ภาพ
                    """
                )


                analyze = st.button(
                    "🔍 วิเคราะห์ภาพ",
                    use_container_width=True
                )


                if analyze:

                    with st.spinner(
                        "🔄 กำลังวิเคราะห์..."
                    ):

                        # Transform image

                        x = transform(
                            image
                        ).unsqueeze(0)

                        x = x.to(device)


                        # Prediction

                        with torch.no_grad():

                            output = model(x)

                            probability = torch.softmax(
                                output,
                                dim=1
                            )[0]


                        # ค่าที่มีความมั่นใจสูงสุด

                        confidence, predicted = torch.max(
                            probability,
                            0
                        )


                        result = class_names[
                            predicted.item()
                        ]


                        confidence_percent = (
                            confidence.item() * 100
                        )


                    # ======================================
                    # RESULT
                    # ======================================

                    st.markdown(
                        """
                        <div class="result-box">
                        """,
                        unsafe_allow_html=True
                    )


                    st.markdown(
                        "## 🌱 ผลการวิเคราะห์"
                    )


                    st.markdown(
                        f"""
                        ### {result}
                        """
                    )


                    st.metric(
                        "🎯 ความมั่นใจ",
                        f"{confidence_percent:.2f}%"
                    )


                    # ======================================
                    # HEALTHY
                    # ======================================

                    if result == "Healthy Banana leaf":

                        st.success(
                            """
                            🌱 ระบบตรวจพบว่า
                            เป็นใบกล้วยปกติ
                            """
                        )


                    # ======================================
                    # DISEASE
                    # ======================================

                    else:

                        st.warning(
                            """
                            ⚠️ ระบบพบลักษณะ
                            ที่อยู่ในกลุ่มโรค
                            หรือความเสียหาย
                            """
                        )


                        st.info(
                            """
                            💡 คำแนะนำ:
                            ควรตรวจสอบใบกล้วยเพิ่มเติม
                            และพิจารณาคำแนะนำจากผู้เชี่ยวชาญ
                            """
                        )


                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )


                    # ======================================
                    # PROBABILITY
                    # ======================================

                    st.markdown(
                        "### 📊 ความน่าจะเป็นของทั้ง 5 Class"
                    )


                    probabilities = (
                        probability
                        .cpu()
                        .numpy()
                        * 100
                    )


                    df = pd.DataFrame({

                        "Class": class_names,

                        "Probability (%)":
                            probabilities

                    })


                    # เรียงจากมากไปน้อย

                    df = df.sort_values(
                        "Probability (%)",
                        ascending=False
                    )


                    # แสดงกราฟ
                    st.bar_chart(
                        df.set_index("Class")
                    )


                    # ตาราง

                    st.dataframe(
                        df.style.format(
                            {
                                "Probability (%)":
                                "{:.2f}%"
                            }
                        ),
                        use_container_width=True,
                        hide_index=True
                    )


                    # ======================================
                    # DISEASE INFO
                    # ======================================

                    st.markdown(
                        "### 📚 ข้อมูล"
                    )


                    st.info(
                        disease_info[result]
                    )


                    # ======================================
                    # NEW ANALYSIS
                    # ======================================

                    st.divider()


                    if st.button(
                        "🔄 เริ่มการวิเคราะห์ใหม่",
                        use_container_width=True
                    ):

                        st.rerun()


# =========================================================
# DISEASE INFORMATION
# =========================================================

elif page == "📚 Disease Information":

    st.markdown(
        """
        <div class="hero">

        <h1>
        📚 Disease Information
        </h1>

        <p>
        ข้อมูลโรคและความเสียหายของใบกล้วย
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    for name in class_names:

        with st.expander(
            f"🌿 {name}"
        ):

            st.markdown(
                f"### {name}"
            )

            st.write(
                disease_info[name]
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    🍌 <b>Banana Leaf AI</b>

    <br>

    Banana Leaf Classification using ResNet50

    <br>

    AI-powered banana leaf analysis

    </div>
    """,
    unsafe_allow_html=True
)
