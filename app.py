import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Banana Leaf AI",
    page_icon="🍌",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #dddddd;
    margin-top: 20px;
}

.info-box {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #dddddd;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# TITLE
# ==================================================

st.markdown(
    '<div class="main-title">🍌 Banana Leaf AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'ระบบจำแนกโรคและความเสียหายของใบกล้วยด้วย ResNet50'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# CLASS NAMES
# ==================================================

class_names = [
    "Banana Skipper Damage",
    "Black and Yellow Sigatoka",
    "Chewing insect damage on banana leaf",
    "Healthy Banana leaf",
    "Panama Wilt Disease"
]


# ==================================================
# DISEASE INFORMATION
# ==================================================

disease_info = {

    "Banana Skipper Damage": {
        "ชื่อ": "ความเสียหายจาก Banana Skipper",
        "สาเหตุ": "เกิดจากแมลงกลุ่ม Banana Skipper ที่เข้ากัดกินใบกล้วย",
        "ลักษณะ": "ใบมีรอยกัดกินหรือส่วนของใบถูกทำลาย",
        "การป้องกัน": "ตรวจสอบใบกล้วยเป็นประจำและควบคุมแมลงอย่างเหมาะสม"
    },

    "Black and Yellow Sigatoka": {
        "ชื่อ": "โรคใบจุดดำและเหลือง",
        "สาเหตุ": "เกิดจากเชื้อราที่ทำให้เกิดรอยโรคบนใบกล้วย",
        "ลักษณะ": "พบจุดหรือรอยแผลสีเข้มและบริเวณสีเหลืองบนใบ",
        "การป้องกัน": "ดูแลความสะอาดของแปลง ตัดใบที่เป็นโรค และเพิ่มการระบายอากาศ"
    },

    "Chewing insect damage on banana leaf": {
        "ชื่อ": "ความเสียหายจากแมลงกัดกินใบ",
        "สาเหตุ": "เกิดจากแมลงที่กัดกินเนื้อใบกล้วย",
        "ลักษณะ": "พบรู รอยแหว่ง หรือบริเวณใบที่ถูกกัดกิน",
        "การป้องกัน": "ตรวจสอบแมลงอย่างสม่ำเสมอและควบคุมแมลงด้วยวิธีที่เหมาะสม"
    },

    "Healthy Banana leaf": {
        "ชื่อ": "ใบกล้วยปกติ",
        "สาเหตุ": "ไม่มีลักษณะโรคหรือความเสียหายตาม Class ที่โมเดลตรวจพบ",
        "ลักษณะ": "ใบมีสภาพสมบูรณ์ ไม่มีรอยโรคหรือรอยกัดกินเด่นชัด",
        "การป้องกัน": "ดูแลน้ำ ธาตุอาหาร แสง และตรวจสอบใบอย่างสม่ำเสมอ"
    },

    "Panama Wilt Disease": {
        "ชื่อ": "โรคตายพราย (Panama Wilt)",
        "สาเหตุ": "เกิดจากเชื้อรา Fusarium ที่สามารถเข้าสู่ระบบท่อลำเลียงของต้นกล้วย",
        "ลักษณะ": "ใบอาจเหลือง เหี่ยว และอาการสามารถรุนแรงขึ้นจนกระทบทั้งต้น",
        "การป้องกัน": "ใช้ต้นพันธุ์ที่สะอาดและระวังการเคลื่อนย้ายดินหรือวัสดุที่อาจปนเปื้อน"
    }
}


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# LOAD MODEL
# ==================================================

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
            "ใน GitHub Repository"
        )

        st.stop()

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model


model = load_model()


# ==================================================
# IMAGE TRANSFORM
# ==================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🍌 Banana Leaf AI")

menu = st.sidebar.radio(
    "เมนู",
    [
        "🏠 หน้าหลัก",
        "🔍 ทำนายใบกล้วย",
        "📚 ข้อมูลโรค"
    ]
)


# ==================================================
# HOME
# ==================================================

if menu == "🏠 หน้าหลัก":

    st.header("🌿 ระบบจำแนกใบกล้วย")

    st.write(
        "ระบบใช้โมเดล Deep Learning ResNet50 "
        "สำหรับจำแนกประเภทของใบกล้วยจากภาพ"
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
            "🖼️ Input",
            "224 × 224"
        )

    st.divider()

    st.subheader("📋 ประเภทที่ระบบสามารถจำแนก")

    for i, name in enumerate(class_names, 1):

        st.write(
            f"**{i}.** {name}"
        )

    st.info(
        "💡 ไปที่เมนู 'ทำนายใบกล้วย' "
        "เพื่ออัปโหลดรูปและให้ AI วิเคราะห์"
    )


# ==================================================
# PREDICTION
# ==================================================

elif menu == "🔍 ทำนายใบกล้วย":

    st.header("🔍 ทำนายประเภทใบกล้วย")

    st.write(
        "อัปโหลดภาพใบกล้วยเพื่อให้ ResNet50 วิเคราะห์"
    )

    uploaded_file = st.file_uploader(
        "📷 เลือกรูปภาพ",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📷 รูปภาพ")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("🤖 การวิเคราะห์")

            if st.button(
                "🔍 วิเคราะห์ภาพ",
                type="primary",
                use_container_width=True
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
                        1
                    )

                predicted_class = class_names[
                    predicted.item()
                ]

                confidence_value = (
                    confidence.item() * 100
                )

                st.success(
                    f"ผลการทำนาย: {predicted_class}"
                )

                st.metric(
                    "ความมั่นใจ",
                    f"{confidence_value:.2f}%"
                )

                st.divider()

                st.subheader(
                    "📊 ความน่าจะเป็นของแต่ละประเภท"
                )

                results = []

                for i, class_name in enumerate(
                    class_names
                ):

                    percent = (
                        probabilities[0][i].item()
                        * 100
                    )

                    results.append(
                        (class_name, percent)
                    )

                results.sort(
                    key=lambda x: x[1],
                    reverse=True
                )

                for class_name, percent in results:

                    st.write(
                        f"**{class_name}** — "
                        f"{percent:.2f}%"
                    )

                    st.progress(
                        int(percent)
                    )

                st.divider()

                st.subheader(
                    "📚 ข้อมูลเพิ่มเติม"
                )

                info = disease_info[
                    predicted_class
                ]

                st.write(
                    f"**ชื่อ:** {info['ชื่อ']}"
                )

                st.write(
                    f"**สาเหตุ:** {info['สาเหตุ']}"
                )

                st.write(
                    f"**ลักษณะ:** {info['ลักษณะ']}"
                )

                st.write(
                    f"**การป้องกัน:** {info['การป้องกัน']}"
                )

                st.warning(
                    "⚠️ ผลนี้เป็นการจำแนกจากโมเดล AI "
                    "ควรใช้ผู้เชี่ยวชาญยืนยันโรคก่อนการจัดการจริง"
                )


# ==================================================
# DISEASE INFORMATION
# ==================================================

else:

    st.header("📚 ข้อมูลโรคและความเสียหาย")

    st.write(
        "เลือกประเภทที่ต้องการดูข้อมูล"
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
        f"🍌 {info['ชื่อ']}"
    )

    st.markdown(
        f"""
        <div class="info-box">

        <h4>🔬 สาเหตุ</h4>

        {info['สาเหตุ']}

        <br><br>

        <h4>👀 ลักษณะ</h4>

        {info['ลักษณะ']}

        <br><br>

        <h4>🛡️ การป้องกัน</h4>

        {info['การป้องกัน']}

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    if selected_class == "Healthy Banana leaf":

        st.success(
            "🌱 ประเภทนี้หมายถึงใบกล้วยที่อยู่ในสภาพปกติ"
        )

    else:

        st.warning(
            "⚠️ ข้อมูลนี้ใช้เพื่อประกอบการเรียนรู้ "
            "ไม่ควรใช้แทนการวินิจฉัยจากผู้เชี่ยวชาญ"
        )
