import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# ตั้งค่าหน้าเว็บ
# ==================================================

st.set_page_config(
    page_title="Banana Leaf AI",
    page_icon="🍌",
    layout="wide"
)


# ==================================================
# CSS ทำให้เว็บดูทันสมัย
# ==================================================

st.markdown("""
<style>

.stApp {
    background: #f5f8f6;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #123c2c;
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* Header */
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
    font-size: 45px;
    margin-bottom: 10px;
    color: #173b2d;
}

.hero p {
    font-size: 18px;
    color: #557065;
}

/* Card */
.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #e2e9e5;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.05);
    min-height: 150px;
}

.card h3 {
    color: #173b2d;
}

.card p {
    color: #687870;
}

/* Result */
.result {
    background: white;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #e0e8e3;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.05);
}

/* Button */
.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    height: 45px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: white;
    padding: 15px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# CLASS
# ==================================================

class_names = [
    "Banana Skipper Damage",
    "Black and Yellow Sigatoka",
    "Chewing insect damage on banana leaf",
    "Healthy Banana leaf",
    "Panama Wilt Disease"
]


# ==================================================
# ข้อมูลโรค
# ==================================================

disease_info = {

    "Banana Skipper Damage":
        "ความเสียหายที่เกิดจากแมลงหรือหนอนที่ทำลายใบกล้วย",

    "Black and Yellow Sigatoka":
        "โรคที่ทำให้เกิดจุดหรือรอยสีเข้มบนใบกล้วย",

    "Chewing insect damage on banana leaf":
        "ความเสียหายจากแมลงที่กัดกินใบ ทำให้เกิดรูหรือรอยแหว่ง",

    "Healthy Banana leaf":
        "ใบกล้วยที่อยู่ในกลุ่มปกติ",

    "Panama Wilt Disease":
        "โรคเหี่ยวของกล้วยที่ส่งผลต่อระบบท่อลำเลียงของต้น"
}


# ==================================================
# MODEL
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "best_resnet50_banana.pth"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


@st.cache_resource
def load_model():

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    if not os.path.exists(MODEL_PATH):
        return None

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    return model


model = load_model()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center">

        <div style="font-size:55px">🍌</div>

        <h2>Banana Leaf AI</h2>

        <p>AI Classification</p>

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

    st.caption(
        "🤖 Model: ResNet50"
    )


# ==================================================
# HOME
# ==================================================

if page == "🏠 Home":

    st.markdown(
        """
        <div class="hero">

        <h1>🍌 Banana Leaf AI</h1>

        <p>
        ระบบ AI สำหรับจำแนกโรคและความเสียหายของใบกล้วย
        ด้วยโมเดล ResNet50
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## 👋 ยินดีต้อนรับ")

    st.write(
        "อัปโหลดภาพใบกล้วยเพื่อให้ระบบ AI วิเคราะห์ "
        "และแสดงประเภทของใบกล้วยพร้อมเปอร์เซ็นต์ความมั่นใจ"
    )

    st.write("")

    # การ์ด
    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="card">

            <h3>📷 วิเคราะห์ภาพ</h3>

            <p>
            อัปโหลดรูปใบกล้วย
            เพื่อให้ AI วิเคราะห์
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "🔍 เริ่มทำนาย",
            use_container_width=True
        ):
            page = "🔍 Predict"
            st.rerun()

    with col2:

        st.markdown(
            """
            <div class="card">

            <h3>🤖 ResNet50</h3>

            <p>
            ใช้โมเดล Deep Learning
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

            <h3>📚 ข้อมูลโรค</h3>

            <p>
            ดูข้อมูลประเภทโรค
            และความเสียหายของใบกล้วย
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("## 🌱 ประเภทที่ระบบสามารถจำแนก")

    for name in class_names:

        st.markdown(
            f"""
            <div class="card" style="margin-bottom:12px">

            <h3>🌿 {name}</h3>

            <p>
            {disease_info[name]}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# ==================================================
# PREDICT
# ==================================================

elif page == "🔍 Predict":

    st.markdown(
        """
        <div class="hero">

        <h1>🔍 วิเคราะห์ใบกล้วย</h1>

        <p>
        อัปโหลดภาพและให้ ResNet50 วิเคราะห์
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    if model is None:

        st.error(
            "ไม่พบไฟล์ best_resnet50_banana.pth"
        )

        st.info(
            "กรุณาใส่ไฟล์โมเดลไว้ใน GitHub "
            "โฟลเดอร์เดียวกับ app.py"
        )

    else:

        uploaded_file = st.file_uploader(
            "📷 เลือกรูปใบกล้วย",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            col1, col2 = st.columns(2)

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
                    "### 🤖 AI Analysis"
                )

                st.write(
                    "กดปุ่มด้านล่างเพื่อเริ่มการวิเคราะห์"
                )

                analyze = st.button(
                    "🔍 วิเคราะห์ภาพ",
                    use_container_width=True
                )

                if analyze:

                    with st.spinner(
                        "กำลังวิเคราะห์..."
                    ):

                        x = transform(
                            image
                        ).unsqueeze(0).to(device)

                        with torch.no_grad():

                            output = model(x)

                            probability = torch.softmax(
                                output,
                                dim=1
                            )[0]

                        confidence, predicted = torch.max(
                            probability,
                            0
                        )

                        result = class_names[
                            predicted.item()
                        ]

                        confidence = (
                            confidence.item() * 100
                        )

                    st.markdown(
                        '<div class="result">',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"## 🌱 {result}"
                    )

                    st.metric(
                        "🎯 ความมั่นใจ",
                        f"{confidence:.2f}%"
                    )

                    if result == "Healthy Banana leaf":

                        st.success(
                            "🌱 ระบบตรวจพบว่าเป็นใบกล้วยปกติ"
                        )

                    else:

                        st.warning(
                            "⚠️ ระบบพบลักษณะที่อยู่ในกลุ่มโรคหรือความเสียหาย"
                        )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                    # กราฟ
                    st.markdown(
                        "### 📊 ความน่าจะเป็นของแต่ละ Class"
                    )

                    probabilities = (
                        probability.cpu().numpy() * 100
                    )

                    df = pd.DataFrame({
                        "Class": class_names,
                        "Probability": probabilities
                    })

                    st.bar_chart(
                        df.set_index("Class")
                    )

                    st.markdown(
                        "### 📚 ข้อมูล"
                    )

                    st.info(
                        disease_info[result]
                    )

                    if st.button(
                        "🔄 วิเคราะห์ภาพใหม่",
                        use_container_width=True
                    ):
                        st.rerun()


# ==================================================
# DISEASE INFORMATION
# ==================================================

elif page == "📚 Disease Information":

    st.markdown(
        """
        <div class="hero">

        <h1>📚 Disease Information</h1>

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


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <br><br>

    <div style="
        text-align:center;
        color:#7b8982;
        padding:20px;
    ">

    🍌 <b>Banana Leaf AI</b><br>
    Banana Leaf Classification using ResNet50

    </div>
    """,
    unsafe_allow_html=True
)
