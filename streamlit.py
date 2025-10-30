import streamlit as st
import torch
from PIL import Image
import pandas as pd
from ultralytics import YOLO

image = st.sidebar.file_uploader("Загрузите фото вашей родинки", type=['jpg', 'png'])

st.title('Детекция кораблей по аэрокосмическим снимкам')

try:
    # Importing model
    model = YOLO("/home/maxim/DS/ds-phase-2/cv_project/yolo8m_model/weights/best.pt")
    
    # Если загружено изображение, используем его, иначе используем путь по умолчанию
    if image is not None:
        # Загружаем изображение через PIL
        input_image = Image.open(image)
        # Конвертируем в RGB если нужно
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        # Делаем предсказание на загруженном изображении
        results = model.predict(
            source=input_image,
            save=False,
            show=False
        )
        
        
        
    else:
        # Используем изображение по умолчанию
        st.write('No image uploaded')
    
    # Выводим результаты предсказания
    if len(results) > 0:
        result = results[0]  # берем первый результат
        
        col1, col2 = st.columns([1, 1])
        # Отображаем оригинальное изображение
        with col1:
            st.image(input_image, caption="Загруженное изображение", use_container_width=True)
        # Отображаем изображение с детекциями
        with col2:
            if hasattr(result, 'plot') and callable(getattr(result, 'plot')):
                plotted_image = result.plot()
                st.image(plotted_image, caption="Результат детекции", use_container_width=True)


        
        
        
        # Выводим информацию о детекциях
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            num_detections = len(boxes)
            
            st.subheader(f"Найдено объектов: {num_detections}")
            
            # Создаем таблицу с результатами
            if num_detections > 0:
                detection_data = []
                for i, box in enumerate(boxes):
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    class_name = result.names[class_id] if hasattr(result, 'names') else f"Class {class_id}"
                    
                    detection_data.append({
                        "Объект": i + 1,
                        "Класс": class_name,
                        "Уверенность": f"{confidence:.3f}"
                    })
                
                # Отображаем таблицу с детекциями
                df = pd.DataFrame(detection_data)
                st.table(df)
                
        else:
            st.write("Объекты не обнаружены")
            
except Exception as e:
    st.write(f"Ошибка: {e}")