import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO


st.set_page_config(page_title="Детекция лиц YOLO", page_icon="👁️", layout="wide")
st.title("Детекция лиц с помощью любой версии YOLOv8 c последующей маскировкой детектированной области")
st.markdown("Модель обучена для распознавания лиц и размывания детектированных областей.")


# Загрузка модели

@st.cache_resource
def load_model():
    model = YOLO(r"models\face_model\train\weights\best.pt")
    return model

model = load_model()


with st.expander("Информация о модели и обучении"):
    st.markdown("""
    **Архитектура:** YOLOv8n
    - **Количество эпох:** 30
    - **Размер выборки:** 16 000 изображений
    - **Метрики:**
    - Precision (P): 0.90
    - Recall (R): 0.78
    - mAP@0.5: 0.855
    - mAP@0.5:0.95: 0.566

    **PR-кривая и confusion matrix:**
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.image(r"models\face_model\train\results.png", caption="Метрики", use_column_width=True)
    with col2:
        st.image(r"models\face_model\train\confusion_matrix.png", caption="Confusion matrix", use_column_width=True)

st.markdown("---")


st.subheader("Загрузите изображения или укажите ссылку")

uploaded_files = st.file_uploader(
    "Выберите одно или несколько изображений",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

url_input = st.text_input("Или введите прямую ссылку на изображение URL:")

images = []

# --- если пользователь ввёл ссылку ---
if url_input:
    try:
        response = requests.get(url_input)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        images.append(image)
    except Exception as e:
        st.error(f"Не удалось загрузить изображение по ссылке: {e}")

# --- если пользователь загрузил файлы ---
for uploaded in uploaded_files:
    image = Image.open(uploaded).convert("RGB")
    images.append(image)


if images:
    st.subheader("Результаты детекции и маскировки:")

    for idx, image in enumerate(images):
        st.markdown(f"### Изображение {idx + 1}")
        img_np = np.array(image)

        # YOLO-предсказание
        results = model(img_np)

        # Размытие детектированных областей
        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                face_region = img_np[y1:y2, x1:x2]
                if face_region.size > 0:
                    face_region = cv2.GaussianBlur(face_region, (51, 51), 30)
                    img_np[y1:y2, x1:x2] = face_region

        # Получаем изображение с рамками
        annotated = results[0].plot()
        # Объединяем размытие и рамки
        blended = cv2.addWeighted(img_np, 0.8, annotated, 0.2, 0)

        # Преобразуем в RGB
        blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

        st.image(blended_rgb, caption="Результат детекции и маскирования", use_column_width=True)

        st.markdown(f"Количество детектированных лиц: **{len(results[0].boxes)}**")
        st.markdown("---")

else:
    st.info("Загрузите изображения или вставьте ссылку, чтобы выполнить детекцию.")
