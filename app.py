# import streamlit as st
# import tempfile
# import os
# import shutil
# from PIL import Image
# import cv2
# import glob
# from ultralytics import YOLO
# import requests
# import pandas as pd
# from pytube import YouTube
# import time
# from moviepy.editor import VideoFileClip

# def predict(source, model="yolov8s-seg.pt", thresh=0.25, save_dir='runs/segment/predict'):
#     # Load a pretrained YOLOv8 model
#     model = YOLO(model)
    
#     # Run inference with arguments
#     results = model(source, save=True, conf=thresh, project=save_dir, name='exp')
#     return results

# def clear_previous_results(directory='runs/segment/predict'):
#     if os.path.exists(directory):
#         for folder in os.listdir(directory):
#             folder_path = os.path.join(directory, folder)
#             if os.path.isdir(folder_path):
#                 shutil.rmtree(folder_path)

# def convert_video_to_mp4(video_path):
#     clip = VideoFileClip(video_path)
#     temp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#     clip.write_videofile(temp_mp4.name)
#     return temp_mp4.name

# def display_results(directory, option, results):
#     if option in ['Image', 'Images and Videos', 'Webcam', 'URL']:
#         files = glob.glob(os.path.join(directory, '*'))
#         # st.write(f"Found files: {files}")
#         if not files:
#             st.error("No files found in the results directory.")
#         for file_path, result in zip(files, results):
#             if file_path.endswith(('.jpg', '.jpeg', '.png')):
#                 st.image(file_path, caption='Processed Image', use_column_width=True)
#                 display_detection_table(result)
#             elif file_path.endswith(('.mp4', '.avi')):
#                 # st.write(f"Displaying video from path: {file_path}")
#                 # Convert to mp4 if necessary
#                 if not file_path.endswith('.mp4'):
#                     file_path = convert_video_to_mp4(file_path)
#                 with open(file_path, "rb") as file:
#                     video_bytes = file.read()
#                     st.video(video_bytes)
#                 display_detection_table(result)
#     elif option == 'Video':
#         video_files = glob.glob(os.path.join(directory, '*.mp4')) + glob.glob(os.path.join(directory, '*.avi'))
#         # st.write(f"Found video files: {video_files}")
#         if not video_files:
#             st.error("No videos found in the results directory.")
#         else:
#             video_file_path = video_files[0]
#             # st.write(f"Displaying video from path: {video_file_path}")
#             # Convert to mp4 if necessary
#             if not video_file_path.endswith('.mp4'):
#                 video_file_path = convert_video_to_mp4(video_file_path)
#             with open(video_file_path, "rb") as file:
#                 video_bytes = file.read()
#                 st.video(video_bytes)
#             display_detection_table(results)

# def capture_and_predict_webcam(model_path, conf_thresh):
#     stframe = st.empty()
#     cap = cv2.VideoCapture(0)
#     model = YOLO(model_path)

#     stop_button = st.sidebar.button("Stop Webcam", key="stop_webcam")

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Failed to capture image from webcam.")
#             break
        
#         # Resize the frame to improve performance
#         frame = cv2.resize(frame, (640, 480))

#         # Run YOLOv8 inference on the frame
#         results = model(frame, conf=conf_thresh)
        
#         # Draw the results on the frame
#         for result in results:
#             annotated_frame = result.plot()
#             annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#             stframe.image(annotated_frame, channels="RGB")

#         if stop_button:
#             break

#         # Add a short delay to improve performance
#         time.sleep(0.1)

#     cap.release()

# def display_detection_table(result):
#     class_counts = {}
#     for detection in result.boxes:
#         class_id = int(detection.cls.item())  # Convert tensor to integer
#         class_name = result.names[class_id]
#         class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
#     if class_counts:
#         df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
#         st.table(df)

# st.title('Instance Segmentation')

# # Sidebar for input options
# st.sidebar.title('Input Options')
# option = st.sidebar.radio(
#     'Select input type:',
#     ('Image', 'Video', 'Webcam', 'Images and Videos', 'URL'),
#     key="input_type_radio"
# )

# # Dropdown for model selection
# model_options = {
#     "main model": "yolov8s-seg.pt",
#     "custom model": "runs\\segment\\train11\\weights\\best.pt"
# }
# selected_model_key = st.sidebar.selectbox('Select model', list(model_options.keys()), key="model_selectbox")
# selected_model = model_options[selected_model_key]

# # Slider for confidence threshold
# confidence_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.25, 0.01, key="confidence_slider")

# source = None
# temp_files = []

# # Handling different input types
# if option == 'Image':
#     uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader")
#     if uploaded_file is not None:
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
#         tfile.write(uploaded_file.read())
#         tfile.close()  # Ensure the file is closed
#         source = tfile.name
#         temp_files.append(tfile.name)

# elif option == 'Video':
#     uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="video_uploader")
#     if uploaded_file is not None:
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#         tfile.write(uploaded_file.read())
#         tfile.close()  # Ensure the file is closed
#         source = tfile.name
#         temp_files.append(tfile.name)

# elif option == 'Webcam':
#     if st.sidebar.button('Start Webcam', key="start_webcam_button"):
#         capture_and_predict_webcam("yolov8s-seg.pt", confidence_threshold)

# elif option == 'Images and Videos':
#     uploaded_files = st.sidebar.file_uploader("Upload files", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"], accept_multiple_files=True, key="multi_uploader")
#     if uploaded_files:
#         temp_dir = tempfile.mkdtemp()
#         for uploaded_file in uploaded_files:
#             temp_path = os.path.join(temp_dir, uploaded_file.name)
#             with open(temp_path, 'wb') as f:
#                 f.write(uploaded_file.read())
#             temp_files.append(temp_path)
#         source = temp_dir

# elif option == 'URL':
#     url = st.sidebar.text_input("Enter image URL", key="url_input")
#     if url:
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
#         with open(temp_file.name, 'wb') as f:
#             f.write(requests.get(url).content)
#         temp_file.close()  # Ensure the file is closed
#         source = temp_file.name
#         temp_files.append(temp_file.name)

# # Run inference if a valid source is provided
# if source is not None and option != 'Webcam':
#     if st.sidebar.button('Run', key="run_button"):
#         with st.spinner('Running...'):
#             # Clear previous results
#             clear_previous_results()
#             results = predict(source, selected_model, confidence_threshold)
#             st.success('Inference completed!')
#             display_results('runs/segment/predict/exp', option, results)

# # Clean up temporary files
# for temp_file in temp_files:
#     try:
#         if os.path.exists(temp_file):
#             os.remove(temp_file)
#     except Exception as e:
#         st.error(f"Error removing temporary file: {e}")
import streamlit as st
import tempfile
import os
import shutil
from PIL import Image
import cv2
import glob
from ultralytics import YOLO
import requests
import pandas as pd
import time
from moviepy.editor import VideoFileClip
from fpdf import FPDF
import plotly.express as px

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Detection Summary", 0, 1, "C")

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(10)

    def chapter_body(self, body):
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 10, body)
        self.ln()

def format_detection_summary(file_path, summary):
    formatted_summary = f"File: {os.path.basename(file_path)}\n"
    class_counts = {}
    for item in summary:
        class_name = item['Class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        bbox = item['Bounding Box'][0]  # Extract the bounding box coordinates
        formatted_summary += (
            f"Class: {class_name}, Confidence: {item['Confidence']:.2f}, "
            f"Bounding Box: [x_min: {bbox[0]:.2f}, y_min: {bbox[1]:.2f}, x_max: {bbox[2]:.2f}, y_max: {bbox[3]:.2f}]\n"
        )
    formatted_summary += "\nClass counts:\n"
    for class_name, count in class_counts.items():
        formatted_summary += f"{class_name}: {count}\n"
    return formatted_summary

def save_to_pdf(detection_summaries, save_path):
    pdf = PDF("L", "mm", "A4")
    for file_path, summary in detection_summaries:
        if summary is not None:  # Only process if summary is not None
            pdf.add_page()
            
            # Add image if it is an image file
            if file_path.endswith(('.jpg', '.jpeg', '.png')):
                pdf.image(file_path, x=10, y=10, w=277)  # Make the image take up the full page width
                pdf.add_page()
            
            formatted_summary = format_detection_summary(file_path, summary)
            pdf.chapter_title(f"File: {os.path.basename(file_path)}")
            pdf.chapter_body(formatted_summary)
    pdf.output(save_path)

def predict(source, model="yolov8s-seg.pt", thresh=0.25, save_dir='runs/segment/predict'):
    start_time = time.time()
    model = YOLO(model)
    results = model(source, save=True, conf=thresh, project=save_dir, name='exp')
    end_time = time.time()
    inference_time = end_time - start_time
    st.write(f"Inference Time: {inference_time:.2f} seconds")
    return results

def clear_previous_results(directory='runs/segment/predict'):
    if os.path.exists(directory):
        for folder in os.listdir(directory):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)

def convert_video_to_mp4(video_path):
    clip = VideoFileClip(video_path)
    temp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    clip.write_videofile(temp_mp4.name)
    return temp_mp4.name

def display_results(directory, option, results):
    detection_summaries = []
    if option in ['Image', 'Images and Videos', 'Webcam', 'URL']:
        files = glob.glob(os.path.join(directory, '*'))
        if not files:
            st.error("No files found in the results directory.")
        for file_path, result in zip(files, results):
            if file_path.endswith(('.jpg', '.jpeg', '.png')):
                st.image(file_path, caption='Processed Image', use_column_width=True)
                summary = display_detection_table(result)
                display_class_counts(summary)
                detection_summaries.append((file_path, summary))
            elif file_path.endswith(('.mp4', '.avi')):
                if not file_path.endswith('.mp4'):
                    file_path = convert_video_to_mp4(file_path)
                with open(file_path, "rb") as file:
                    video_bytes = file.read()
                    st.video(video_bytes)
                detection_summaries.append((file_path, None))
    elif option == 'Video':
        video_files = glob.glob(os.path.join(directory, '*.mp4')) + glob.glob(os.path.join(directory, '*.avi'))
        if not video_files:
            st.error("No videos found in the results directory.")
        else:
            video_file_path = video_files[0]
            if not video_file_path.endswith('.mp4'):
                video_file_path = convert_video_to_mp4(video_file_path)
            with open(video_file_path, "rb") as file:
                video_bytes = file.read()
                st.video(video_bytes)
            detection_summaries.append((video_file_path, None))
    
    return detection_summaries

def capture_and_predict_webcam(model_path, conf_thresh):
    # Initialize the YOLO model
    model = YOLO(model_path)
    
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    # Create a button to stop the webcam
    stop_button = st.sidebar.button("Stop Webcam", key="stop_webcam")

    # Frame display container
    stframe = st.empty()

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break
        
        # Resize the frame to improve performance
        frame = cv2.resize(frame, (640, 480))
        
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=conf_thresh)

        # Annotate the frame with the results
        annotated_frame = results[0].plot()
        
        # Convert annotated frame to RGB
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the resulting frame in Streamlit
        stframe.image(annotated_frame_rgb, channels="RGB")
        
        # Check if the stop button has been pressed
        if stop_button:
            break

        # Add a short delay to improve performance
        time.sleep(0.1)

    # When everything is done, release the capture
    cap.release()


def display_detection_table(result):
    detection_data = []
    class_counts = {}
    for detection in result.boxes:
        class_id = int(detection.cls.item())  # Convert tensor to integer
        class_name = result.names[class_id]
        confidence = detection.conf.item()  # Get confidence score
        bbox = detection.xyxy.tolist()  # Get bounding box coordinates
        detection_data.append({
            "Class": class_name,
            "Confidence": confidence,
            "Bounding Box": bbox
        })
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    if detection_data:
        df = pd.DataFrame(detection_data)
        # Plotting
        fig = px.bar(df, x="Class", y="Confidence", title="Confidence Scores for Detected Classes")
        st.plotly_chart(fig)
        
        return detection_data

def display_class_counts(summary):
    class_counts = {}
    for item in summary:
        class_name = item['Class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    if class_counts:
        df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
        st.table(df)

st.title('Instance Segmentation')

# Sidebar for input options
st.sidebar.title('Input Options')
option = st.sidebar.radio(
    'Select input type:',
    ('Image', 'Video', 'Webcam', 'Images and Videos', 'URL'),
    key="input_type_radio"
)

# Dropdown for model selection
model_options = {
    "YOLOv8n-seg": "models/yolov8n-seg.pt",  # Lightweight and fast, lower accuracy
    "YOLOv8s-seg": "models/yolov8s-seg.pt",  # Small, balanced speed and accuracy
    "YOLOv8m-seg": "models/yolov8m-seg.pt",  # Medium, higher accuracy, slower than s
    "YOLOv8l-seg": "models/yolov8l-seg.pt",  # Large, higher accuracy, slower
    "YOLOv8x-seg": "models/yolov8x-seg.pt",  # Extra-large, highest accuracy, slowest
    "custom model": "runs\\segment\\train11\\weights\\best.pt",
}
selected_model_key = st.sidebar.selectbox('Select model', list(model_options.keys()), key="model_selectbox")
selected_model = model_options[selected_model_key]

# Slider for confidence threshold
confidence_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.25, 0.01, key="confidence_slider")

source = None
temp_files = []

# Handling different input types
if option == 'Image':
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader")
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        tfile.write(uploaded_file.read())
        tfile.close()  # Ensure the file is closed
        source = tfile.name
        temp_files.append(tfile.name)

elif option == 'Video':
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="video_uploader")
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()  # Ensure the file is closed
        source = tfile.name
        temp_files.append(tfile.name)

elif option == 'Webcam':
    if st.sidebar.button('Start Webcam', key="start_webcam_button"):
        capture_and_predict_webcam(selected_model, confidence_threshold)

elif option == 'Images and Videos':
    uploaded_files = st.sidebar.file_uploader("Upload files", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"], accept_multiple_files=True, key="multi_uploader")
    if uploaded_files:
        temp_dir = tempfile.mkdtemp()
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            temp_files.append(temp_path)
        source = temp_dir

elif option == 'URL':
    url = st.sidebar.text_input("Enter image URL", key="url_input")
    if url:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        with open(temp_file.name, 'wb') as f:
            f.write(requests.get(url).content)
        temp_file.close()  # Ensure the file is closed
        source = temp_file.name
        temp_files.append(temp_file.name)

# Run inference if a valid source is provided
if source is not None and option != 'Webcam':
    if st.sidebar.button('Run', key="run_button"):
        with st.spinner('Running...'):
            clear_previous_results()
            results = predict(source, selected_model, confidence_threshold)
            st.success('Inference completed!')
            detection_summaries = display_results('runs/segment/predict/exp', option, results)
            # Save results to PDF
            pdf_path = os.path.join('runs/segment/predict', 'detection_summary.pdf')
            save_to_pdf(detection_summaries, pdf_path)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(label="Download Report", data=pdf_file, file_name="detection_summary.pdf", key="download_pdf_button")

# Clean up temporary files
for temp_file in temp_files:
    try:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    except Exception as e:
        st.error(f"Error removing temporary file: {e}")
