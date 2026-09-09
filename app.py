import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd


# =========================================================
# 1. ตั้งค่าหน้าเว็บ
# =========================================================

st.set_page_config(
    page_title="Banana Leaf AI",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CSS สำหรับ GUI และมือถือ
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.3);
    margin-top: 20px;
}

.info-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.3);
    margin-top: 15px;
}

.big-result {
    font-size: 28px;
    font-weight: 700;
}

.small-text {
    font-size: 15px;
}

@media (max-width: 768px) {

    .title {
        font-size: 30px;
    }

    .subtitle {
        font-size: 15px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .big-result {
        font-size: 22px;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. ชื่อ Class
# =========================================================

class_names = [
    "Banana Skipper Damage",
    "Black and Yellow Sigatoka",
    "Chewing insect damage on banana leaf",
    "Healthy Banana leaf",
    "Panama Wilt Disease"
]


# =========================================================
# 4. ข้อมูลโรค
# =========================================================

disease_info = {

    "Banana Skipper Damage": {
        "ชื่อไทย": "ความเสียหายจาก Banana Skipper",
        "สาเหตุ": "เกิดจากแมลงกลุ่ม Banana Skipper ที่กัดกินใบกล้วย",
        "ลักษณะ": "ใบกล้วยมีรอยกัดกินหรือบริเวณเนื้อใบถูกทำลาย",
        "การป้องกัน": "ตรวจสอบใบกล้วยเป็นประจำและควบคุมแมลงอย่างเหมาะสม",
        "คำแนะนำ": "ควรตรวจสอบบริเวณใบและต้นกล้วยโดยรอบเพื่อประเมินความเสียหาย"
    },

    "Black and Yellow Sigatoka": {
        "ชื่อไทย": "โรคใบจุดดำและเหลือง",
        "สาเหตุ": "เกิดจากเชื้อราที่ทำให้เกิดรอยโรคบนใบกล้วย",
        "ลักษณะ": "พบจุดหรือรอยแผลสีเข้มและบริเวณสีเหลืองบนใบ",
        "การป้องกัน": "ดูแลความสะอาดของแปลง ตัดใบที่เป็นโรค และดูแลการระบายอากาศ",
        "คำแนะนำ": "หากพบอาการเพิ่มขึ้น ควรแยกใบที่เสียหายและขอคำแนะนำจากผู้เชี่ยวชาญ"
    },

    "Chewing insect damage on banana leaf": {
        "ชื่อไทย": "ความเสียหายจากแมลงกัดกินใบ",
        "สาเหตุ": "เกิดจากแมลงที่กัดกินเนื้อใบกล้วย",
        "ลักษณะ": "พบรู รอยแหว่ง หรือบริเวณใบที่ถูกกัดกิน",
        "การป้องกัน": "ตรวจสอบแมลงอย่างสม่ำเสมอและควบคุมแมลงด้วยวิธีที่เหมาะสม",
        "คำแนะนำ": "ควรตรวจสอบทั้งด้านบนและด้านล่างของใบเพื่อค้นหาแมลงหรือร่องรอยการกัดกิน"
    },

    "Healthy Banana leaf": {
        "ชื่อไทย": "ใบกล้วยปกติ",
        "สาเหตุ": "ไม่พบลักษณะความเสียหายตามประเภทที่โมเดลจำแนก",
        "ลักษณะ": "ใบมีสภาพสมบูรณ์และไม่พบรอยโรคหรือรอยกัดกินที่เด่นชัด",
        "การป้องกัน": "ดูแลน้ำ ธาตุอาหาร แสง และตรวจสอบใบอย่างสม่ำเสมอ",
        "คำแนะนำ": "ควรดูแลต้นกล้วยอย่างต่อเนื่องและตรวจสอบใบเป็นประจำ"
    },

    "Panama Wilt Disease": {
        "ชื่อไทย": "โรคตายพราย (Panama Wilt)",
        "สาเหตุ": "เกิดจากเชื้อรา Fusarium ที่สามารถเข้าสู่ระบบท่อลำเลียงของต้นกล้วย",
        "ลักษณะ": "ใบอาจเหลือง เหี่ยว และอาการสามารถรุนแรงขึ้นจนกระทบทั้งต้น",
        "การป้องกัน": "ใช้ต้นพันธุ์ที่สะอาดและระวังการเคลื่อนย้ายดินหรือวัสดุที่อาจปนเปื้อน",
        "คำแนะนำ": "หากสงสัยว่าเป็นโรค ควรแยกต้นที่มีอาการและปรึกษาผู้เชี่ยวชาญ"
    }
}


# =========================================================
# 5. Device
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# 6. โหลด ResNet50
# =========================================================

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
            "❌ ไม่พบไฟล์ best_resnet50_banana.pth "
            "กรุณาตรวจสอบว่าไฟล์โมเดลอยู่ใน GitHub Repository"
        )

        st.stop()

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    # รองรับทั้ง state_dict ปกติ
    # และ checkpoint ที่มี state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


model = load_model()


# =========================================================
# 7. Image Transform
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
# 8. Header
# =========================================================

st.markdown(
    '<div class="title">🍌 Banana Leaf AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'ระบบจำแนกโรคและความเสียหายของใบกล้วยด้วย ResNet50'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# 9. Sidebar Menu
# =========================================================

st.sidebar.title("🍌 Banana Leaf AI")

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "เลือกเมนู",
    [
        "🏠 Home",
        "🔍 Predict",
        "📚 Disease Information"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "ระบบนี้ใช้โมเดล ResNet50 "
    "สำหรับจำแนกภาพใบกล้วยจำนวน 5 Class"
)


# =========================================================
# 10. HOME
# =========================================================

if menu == "🏠 Home":

    st.header("🏠 หน้าหลัก")

    st.write(
        "ระบบ Banana Leaf AI เป็นระบบจำแนกภาพใบกล้วย "
        "โดยใช้โมเดล Deep Learning ResNet50"
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🤖 Model",
            "ResNet50"
        )

    with col2:

        st.metric(
            "🌿 Classes",
            "5"
        )

    with col3:

        st.metric(
            "🖼️ Image Size",
            "224 × 224"
        )

    st.divider()

    st.subheader("🌿 ประเภทที่ระบบสามารถจำแนก")

    for i, name in enumerate(class_names, 1):

        st.write(
            f"**{i}.** {name}"
        )

    st.divider()

    st.subheader("📌 วิธีใช้งาน")

    st.write("1. ไปที่เมนู **🔍 Predict**")

    st.write("2. อัปโหลดรูปใบกล้วย")

    st.write("3. กดปุ่ม **🔍 วิเคราะห์ภาพ**")

    st.write("4. ระบบจะใช้ ResNet50 วิเคราะห์ภาพ")

    st.write("5. ดูผลการทำนายและเปอร์เซ็นต์ความมั่นใจ")

    st.write("6. สามารถดูข้อมูลของประเภทที่ตรวจพบได้")


# =========================================================
# 11. PREDICT
# =========================================================

elif menu == "🔍 Predict":

    st.header("🔍 วิเคราะห์ใบกล้วย")

    st.write(
        "อัปโหลดรูปใบกล้วยเพื่อให้ระบบ ResNet50 วิเคราะห์"
    )

    uploaded_file = st.file_uploader(
        "📷 เลือกรูปภาพ",
        type=["jpg", "jpeg", "png"],
        key="banana_image"
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns(
            [1, 1],
            gap="large"
        )

        with col1:

            st.subheader("📷 รูปภาพที่อัปโหลด")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("🔍 การวิเคราะห์")

            analyze_button = st.button(
                "🔍 วิเคราะห์ภาพ",
                type="primary",
                use_container_width=True
            )

            reset_button = st.button(
                "🔄 เริ่มการวิเคราะห์ใหม่",
                use_container_width=True
            )

        # =================================================
        # RESET
        # =================================================

        if reset_button:

            st.session_state.clear()

            st.rerun()

        # =================================================
        # PREDICTION
        # =================================================

        if analyze_button:

            with st.spinner(
                "🤖 ResNet50 กำลังวิเคราะห์..."
            ):

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
                        dim=1
                    )

            predicted_class = class_names[
                predicted.item()
            ]

            confidence_value = (
                confidence.item() * 100
            )

            # เก็บผลไว้ใน Session
            st.session_state["predicted_class"] = predicted_class
            st.session_state["confidence"] = confidence_value
            st.session_state["probabilities"] = probabilities[0].cpu().tolist()

        # =================================================
        # DISPLAY RESULT
        # =================================================

        if "predicted_class" in st.session_state:

            predicted_class = st.session_state[
                "predicted_class"
            ]

            confidence_value = st.session_state[
                "confidence"
            ]

            probability_values = st.session_state[
                "probabilities"
            ]

            st.divider()

            st.subheader("🤖 ผลการทำนาย")

            st.markdown(
                f"""
                <div class="result-card">

                <div class="big-result">
                {predicted_class}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "📊 ความมั่นใจของโมเดล",
                f"{confidence_value:.2f}%"
            )

            # =================================================
            # HEALTHY / DISEASE
            # =================================================

            if predicted_class == "Healthy Banana leaf":

                st.success(
                    "🌱 ใบกล้วยปกติ — "
                    "ไม่พบลักษณะความเสียหายตาม Class ที่โมเดลจำแนก"
                )

            else:

                st.warning(
                    "⚠️ ระบบตรวจพบประเภทที่อาจมีความผิดปกติ "
                    "ควรตรวจสอบเพิ่มเติม"
                )

            # =================================================
            # PROBABILITY GRAPH
            # =================================================

            st.divider()

            st.subheader(
                "📈 กราฟความน่าจะเป็นทั้ง 5 Class"
            )

            chart_data = pd.DataFrame({

                "ประเภท": class_names,

                "ความน่าจะเป็น (%)": [
                    value * 100
                    for value in probability_values
                ]

            })

            chart_data = chart_data.sort_values(
                "ความน่าจะเป็น (%)",
                ascending=False
            )

            st.bar_chart(
                chart_data.set_index("ประเภท")
            )

            # =================================================
            # ALL PROBABILITIES
            # =================================================

            st.subheader(
                "📊 รายละเอียดความน่าจะเป็น"
            )

            for i, class_name in enumerate(
                class_names
            ):

                percent = (
                    probability_values[i] * 100
                )

                st.write(
                    f"**{class_name}** — "
                    f"{percent:.2f}%"
                )

                st.progress(
                    min(int(percent), 100)
                )

            # =================================================
            # DISEASE INFORMATION
            # =================================================

            st.divider()

            st.subheader(
                "📚 ข้อมูลของผลการทำนาย"
            )

            info = disease_info[
                predicted_class
            ]

            st.write(
                f"### {info['ชื่อไทย']}"
            )

            st.write(
                f"**🔬 สาเหตุ:** {info['สาเหตุ']}"
            )

            st.write(
                f"**👀 ลักษณะ:** {info['ลักษณะ']}"
            )

            st.write(
                f"**🛡️ การป้องกัน:** {info['การป้องกัน']}"
            )

            st.write(
                f"**💡 คำแนะนำ:** {info['คำแนะนำ']}"
            )

            # =================================================
            # WARNING
            # =================================================

            if predicted_class != "Healthy Banana leaf":

                st.warning(
                    "⚠️ ผลการทำนายเป็นผลจากโมเดล AI "
                    "ควรใช้ผู้เชี่ยวชาญยืนยันก่อนดำเนินการจริง"
                )


# =========================================================
# 12. DISEASE INFORMATION
# =========================================================

else:

    st.header(
        "📚 Disease Information"
    )

    st.write(
        "เลือกประเภทโรคหรือความเสียหาย "
        "เพื่อดูข้อมูลเพิ่มเติม"
    )

    selected_class = st.selectbox(
        "🌿 เลือกประเภท",
        class_names
    )

    info = disease_info[
        selected_class
    ]

    st.divider()

    st.subheader(
        f"🍌 {info['ชื่อไทย']}"
    )

    st.markdown(
        f"""
        <div class="info-card">

        <h4>🔬 สาเหตุ</h4>

        {info['สาเหตุ']}

        <br><br>

        <h4>👀 ลักษณะ</h4>

        {info['ลักษณะ']}

        <br><br>

        <h4>🛡️ การป้องกัน</h4>

        {info['การป้องกัน']}

        <br><br>

        <h4>💡 คำแนะนำ</h4>

        {info['คำแนะนำ']}

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    if selected_class == "Healthy Banana leaf":

        st.success(
            "🌱 ใบกล้วยปกติ"
        )

    else:

        st.warning(
            "⚠️ ข้อมูลนี้ใช้เพื่อการเรียนรู้และประกอบการประเมินเบื้องต้น "
            "ควรปรึกษาผู้เชี่ยวชาญเพื่อยืนยันโรค"
        )
