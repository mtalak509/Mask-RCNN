# import streamlit as st
# import torch
# from PIL import Image
# import pandas as pd
# from ultralytics import YOLO
# import matplotlib.pyplot as plt

# image = st.sidebar.file_uploader("Загрузите фото вашей родинки", type=['jpg', 'png'])

# st.title('Детекция кораблей по аэрокосмическим снимкам')

# try:
#     # Importing model
#     model = YOLO("models/ships_yolo8/weights/best.pt")
#     results = pd.read_csv('models/ships_yolo8/results/results.csv')
    
#     # Если загружено изображение, используем его, иначе используем путь по умолчанию
#     if image is not None:
#         # Загружаем изображение через PIL
#         input_image = Image.open(image)
        
#         # Делаем предсказание на загруженном изображении
#         results = model.predict(
#             source=input_image,
#             save=False,
#             show=False
#         )
        
#     else:
#         # Используем изображение по умолчанию
#         st.write('No image uploaded')
    
#     # Выводим результаты предсказания
#     if len(results) > 0:
#         result = results[0]  # берем первый результат
        
#         col1, col2 = st.columns([1, 1])
#         # Отображаем оригинальное изображение
#         with col1:
#             st.image(input_image, caption="Загруженное изображение", use_container_width=True)
#         # Отображаем изображение с детекциями
#         with col2:
#             if hasattr(result, 'plot') and callable(getattr(result, 'plot')):
#                 plotted_image = result.plot()
#                 st.image(plotted_image, caption="Результат детекции", use_container_width=True)


#         # Выводим информацию о детекциях
#         if hasattr(result, 'boxes') and result.boxes is not None:
#             boxes = result.boxes
#             num_detections = len(boxes)
            
#             st.subheader(f"Найдено объектов: {num_detections}")
            
#             # Создаем таблицу с результатами
#             if num_detections > 0:
#                 detection_data = []
#                 for i, box in enumerate(boxes):
#                     confidence = box.conf.item()
#                     class_id = int(box.cls.item())
#                     class_name = result.names[class_id] if hasattr(result, 'names') else f"Class {class_id}"
                    
#                     detection_data.append({
#                         "Объект": i + 1,
#                         "Класс": class_name,
#                         "Уверенность": f"{confidence:.3f}"
#                     })
                
#                 # Отображаем таблицу с детекциями
#                 df = pd.DataFrame(detection_data)
#                 st.table(df)

#         else:
#             st.write("Объекты не обнаружены")


#     def plot_training_results(df):
#         """Простая функция для отрисовки графиков обучения из results.csv"""
        
#         plt.figure(figsize=(10, 5))

#         plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', marker='o')
#         plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', marker='o')
#         plt.xlabel('Epoch')
#         plt.ylabel('mAP')
#         plt.title('mAP Metrics Over Time')
#         plt.legend()
#         plt.grid(True)
#         plt.show()
        
#         # 2. Precision и Recall
#         plt.figure(figsize=(10, 5))
#         plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', marker='o')
#         plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', marker='o')
#         plt.xlabel('Epoch')
#         plt.ylabel('Score')
#         plt.title('Precision & Recall Over Time')
#         plt.legend()
#         plt.grid(True)
#         plt.show()
        
#         # 3. Потери
#         plt.figure(figsize=(10, 5))
#         plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss', marker='o')
#         plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', marker='o')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.title('Training Loss Over Time')
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#         # Использование
#     plot_training_results(results)

# except Exception as e:
#     st.write(f"Ошибка: {e}")








import streamlit as st
import torch
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# Настройка страницы
st.set_page_config(page_title="Детекция кораблей", layout="wide")

# Сайдбар для загрузки изображения
st.sidebar.title("Настройки")
image = st.sidebar.file_uploader("Загрузите аэрокосмический снимок", type=['jpg', 'png', 'jpeg'])

st.title('🚢 Детекция кораблей по аэрокосмическим снимкам')

try:
    # Загрузка модели
    model = YOLO("models/ships_yolo8/weights/best.pt")
    results_df = pd.read_csv('models/ships_yolo8/results/results.csv')
    
    # Основной контент
    if image is not None:
        # Загружаем изображение через PIL
        input_image = Image.open(image)
        
        # Делаем предсказание на загруженном изображении
        with st.spinner('Обрабатываем изображение...'):
            detection_results = model.predict(
                source=input_image,
                save=False,
                conf=0.25,
                imgsz=640
            )
        
        # Выводим результаты предсказания
        if len(detection_results) > 0:
            result = detection_results[0]  # берем первый результат
            
            col1, col2 = st.columns([1, 1])
            
            # Отображаем оригинальное изображение
            with col1:
                st.image(input_image, caption="Загруженное изображение", use_container_width=True)
            
            # Отображаем изображение с детекциями
            with col2:
                if hasattr(result, 'plot') and callable(getattr(result, 'plot')):
                    plotted_image = result.plot()
                    # Конвертируем BGR в RGB для правильного отображения
                    plotted_image_rgb = plotted_image[:, :, ::-1]  # BGR to RGB
                    st.image(plotted_image_rgb, caption="Результат детекции", use_container_width=True)

            # Выводим информацию о детекциях
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                num_detections = len(boxes)
                
                st.subheader(f"🎯 Найдено кораблей: {num_detections}")
                
                # Создаем таблицу с результатами
                if num_detections > 0:
                    detection_data = []
                    for i, box in enumerate(boxes):
                        confidence = box.conf.item()
                        class_id = int(box.cls.item())
                        class_name = result.names[class_id] if hasattr(result, 'names') else f"Class {class_id}"
                        
                        detection_data.append({
                            "Корабль №": i + 1,
                            "Класс": class_name,
                            "Уверенность": f"{confidence:.3f}"
                        })
                    
                    # Отображаем таблицу с детекциями
                    df_detections = pd.DataFrame(detection_data)
                    st.dataframe(df_detections, use_container_width=True)
                    
                    # Статистика
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Всего обнаружено", num_detections)
                    with col_stat2:
                        avg_confidence = np.mean([box.conf.item() for box in boxes])
                        st.metric("Средняя уверенность", f"{avg_confidence:.3f}")
                    with col_stat3:
                        max_confidence = np.max([box.conf.item() for box in boxes])
                        st.metric("Макс. уверенность", f"{max_confidence:.3f}")

            else:
                st.warning("Корабли не обнаружены")
    
    else:
        st.info("👆 Пожалуйста, загрузите аэрокосмический снимок для детекции кораблей")
        
        # Показываем пример работы
        st.subheader("Пример работы модели")
        st.image("https://via.placeholder.com/600x400/4F8BF9/FFFFFF?text=Загрузите+изображение+кораблей", 
                caption="Здесь будет результат детекции после загрузки изображения", 
                use_container_width=True)

    # Вкладки для метрик и графиков
    st.markdown("---")
    st.header("📊 Метрики обучения модели")
    
    tab1, tab2, tab3 = st.tabs(["Графики обучения", "Статистика", "Лучшие результаты"])
    
    with tab1:
        st.subheader("Графики процесса обучения")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # График mAP метрик
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(results_df['epoch'], results_df['metrics/mAP50(B)'], label='mAP50', marker='o', linewidth=2)
            ax1.plot(results_df['epoch'], results_df['metrics/mAP50-95(B)'], label='mAP50-95', marker='o', linewidth=2)
            ax1.set_xlabel('Эпоха')
            ax1.set_ylabel('mAP')
            ax1.set_title('mAP метрики')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
        
        with col2:
            # График Precision и Recall
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(results_df['epoch'], results_df['metrics/precision(B)'], label='Precision', marker='o', linewidth=2, color='green')
            ax2.plot(results_df['epoch'], results_df['metrics/recall(B)'], label='Recall', marker='o', linewidth=2, color='orange')
            ax2.set_xlabel('Эпоха')
            ax2.set_ylabel('Score')
            ax2.set_title('Precision & Recall')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # График потерь
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.plot(results_df['epoch'], results_df['train/box_loss'], label='Box Loss', marker='o', linewidth=2)
            ax3.plot(results_df['epoch'], results_df['train/cls_loss'], label='Class Loss', marker='o', linewidth=2)
            ax3.plot(results_df['epoch'], results_df['train/dfl_loss'], label='DFL Loss', marker='o', linewidth=2)
            ax3.set_xlabel('Эпоха')
            ax3.set_ylabel('Loss')
            ax3.set_title('Training Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
        
        with col4:
            # График Learning Rate
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            ax4.plot(results_df['epoch'], results_df['lr/pg0'], label='Learning Rate', marker='o', linewidth=2, color='purple')
            ax4.set_xlabel('Эпоха')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')
            st.pyplot(fig4)
    
    with tab2:
        st.subheader("Статистика обучения")
        
        # Лучшие метрики
        best_epoch_map50 = results_df['metrics/mAP50(B)'].idxmax()
        best_map50 = results_df['metrics/mAP50(B)'].max()
        best_map = results_df['metrics/mAP50-95(B)'].max()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Лучшая mAP50", f"{best_map50:.4f}")
        with col2:
            st.metric("Эпоха лучшей mAP50", best_epoch_map50 + 1)
        with col3:
            st.metric("Лучшая mAP50-95", f"{best_map:.4f}")
        with col4:
            final_map50 = results_df['metrics/mAP50(B)'].iloc[-1]
            st.metric("Финальная mAP50", f"{final_map50:.4f}")
        
        # Таблица с основными метриками
        st.subheader("Метрики по эпохам")
        display_columns = ['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 
                          'metrics/precision(B)', 'metrics/recall(B)']
        display_df = results_df[display_columns].copy()
        display_df.columns = ['Эпоха', 'mAP50', 'mAP50-95', 'Precision', 'Recall']
        st.dataframe(display_df.round(4), use_container_width=True)
    
    with tab3:
        st.subheader("Лучшие результаты обучения")
        
        # Находим лучшие эпохи для каждой метрики
        best_metrics = {
            'mAP50': results_df.loc[results_df['metrics/mAP50(B)'].idxmax()],
            'mAP50-95': results_df.loc[results_df['metrics/mAP50-95(B)'].idxmax()],
            'Precision': results_df.loc[results_df['metrics/precision(B)'].idxmax()],
            'Recall': results_df.loc[results_df['metrics/recall(B)'].idxmax()],
        }
        
        for metric_name, best_row in best_metrics.items():
            with st.expander(f"Лучшая {metric_name}"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**Эпоха:** {int(best_row['epoch']) + 1}")
                with col2:
                    st.write(f"**{metric_name}:** {best_row[f'metrics/{metric_name}(B)']:.4f}")
                with col3:
                    st.write(f"**Precision:** {best_row['metrics/precision(B)']:.4f}")
                with col4:
                    st.write(f"**Recall:** {best_row['metrics/recall(B)']:.4f}")

except Exception as e:
    st.error(f"❌ Произошла ошибка: {e}")
    st.info("Пожалуйста, проверьте пути к файлам модели и результатов")