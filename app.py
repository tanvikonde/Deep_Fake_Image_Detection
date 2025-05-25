# app.py
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import deque

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths (placeholders for you to fill)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_detection_model.keras")
COVER_IMAGE = os.path.join(BASE_DIR, "cover_image.png")  # Add your own
ACCURACY_PLOT = os.path.join(BASE_DIR, "accuracy_plot.png")  # Add your own
LOSS_PLOT = os.path.join(BASE_DIR, "loss_plot.png")  # Add your own
CONFUSION_PLOT = os.path.join(BASE_DIR, "confusion_matrix.png")  # Add your own

# Load model
@st.cache_resource
def load_detection_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please ensure 'deepfake_detection_model.keras' is in the directory.")
        return None
    try:
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_detection_model()

# Preprocess image
def preprocess_image(image, target_size=(96, 96)):
    try:
        image = cv2.resize(image, target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

# Predict with heatmap
def predict_image(image):
    processed_image = preprocess_image(image)
    if processed_image is None or model is None:
        return None, None, None
    prediction = model.predict(processed_image)
    confidence = np.max(prediction) * 100
    class_label = np.argmax(prediction, axis=1)[0]
    result = "Fake" if class_label == 0 else "Real"
    heatmap = np.mean(processed_image[0], axis=2)
    return result, confidence, heatmap

# Custom UI CSS (Innovative and Professional Design)
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0a0f1f 0%, #1e2a44 100%);
        padding: 2rem;
        font-family: 'Poppins', sans-serif;
        color: #d9e1e8;
    }
    .title {
        font-size: 2.4rem;
        font-weight: 600;
        color: #5e81ac;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 2px solid #81a1c1;
        padding-bottom: 0.5rem;
    }
    .card {
        background: #1e2a44;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
        border: 1px solid #2e3b55;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(94, 129, 172, 0.25);
    }
    .result-fake {
        color: #bf616a;
        font-weight: bold;
        text-align: center;
    }
    .result-real {
        color: #88c0d0;
        font-weight: bold;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: #1e2a44;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        padding: 1rem;
        color: #d9e1e8;
    }
    .webcam-card {
        background: #1e2a44;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
        border: 1px solid #2e3b55;
    }
    .webcam-overlay {
        background: rgba(30, 42, 68, 0.9);
        padding: 10px 15px;
        border-radius: 8px;
        color: #d9e1e8;
        font-size: 1.1rem;
        font-weight: 500;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        margin-top: 10px;
        text-align: center;
    }
    h2, h3 {
        color: #5e81ac;
    }
    .status-bar {
        background: #2e3b55;
        padding: 0.5rem;
        border-radius: 8px;
        color: #d9e1e8;
        font-size: 0.9rem;
        text-align: center;
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Main app layout
def main():
    st.markdown('<h1 class="title">Deepfake Detection & Prevention System</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        if os.path.exists(COVER_IMAGE):
            st.image(COVER_IMAGE, use_container_width=True)
        else:
            st.markdown('<div class="card" style="text-align:center;padding:1rem;">Add your cover image here</div>', unsafe_allow_html=True)
        st.markdown("### Navigation", unsafe_allow_html=True)
        page = st.radio("", ["Home", "Detection", "About", "Model Stats", "Live Demo"], key="nav")

    # Page content
    if page == "Home":
        st.markdown("""
        <div class="card">
            <h2>Next-Gen Media Authentication</h2>
            <p>An advanced AI-driven platform to detect and prevent deepfake manipulations, ensuring trust and security in digital content.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="card">
                <h3>Innovative Features</h3>
                <ul>
                    <li>Real-Time Deepfake Detection</li>
                    <li>Confidence Trend Analysis</li>
                    <li>Attention Heatmap Insights</li>
                    <li>Batch Processing Capabilities</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="card">
                <h3>Impact Areas</h3>
                <ul>
                    <li>Digital Forensics</li>
                    <li>Media Integrity</li>
                    <li>Cybersecurity Defense</li>
                    <li>Public Trust Enhancement</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    elif page == "Detection":
        st.markdown('<h2 class="title">Image Analysis</h2>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], 
                                        accept_multiple_files=True, help="Supports multiple images")
        
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f"Analyzing {uploaded_file.name}..."):
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, 1)
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(image, channels="BGR", caption=f"Image: {uploaded_file.name}")
                    
                    with col2:
                        result, confidence, heatmap = predict_image(image)
                        if result:
                            result_class = "result-fake" if result == "Fake" else "result-real"
                            st.markdown(f"""
                            <div class="card">
                                <h3 class="{result_class}">Prediction: {result}</h3>
                                <p>Confidence: {confidence:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=confidence,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Confidence Score"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#88c0d0" if result == "Real" else "#bf616a", 'thickness': 0.2},
                                    'bgcolor': "#2e3b55",
                                    'borderwidth': 1,
                                    'bordercolor': "#d9e1e8"
                                }
                            ))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            fig, ax = plt.subplots()
                            sns.heatmap(heatmap, cmap="coolwarm", ax=ax, cbar_kws={'label': 'Attention Intensity'})
                            st.pyplot(fig)
                
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.markdown(f'<div class="status-bar">Processed {i + 1} of {len(uploaded_files)} images</div>', unsafe_allow_html=True)

    elif page == "About":
        st.markdown("""
        <div class="card">
            <h2>About the System</h2>
            <p>This system leverages cutting-edge deep learning to combat deepfakes by analyzing subtle inconsistencies in media, offering a robust defense against digital fraud.</p>
            <h3>How It Works</h3>
            <details>
                <summary style="color: #88c0d0; cursor: pointer;">Click to Expand</summary>
                <ul>
                    <li><b>Feature Extraction:</b> Identifies facial distortions and blending artifacts.</li>
                    <li><b>Confidence Scoring:</b> Quantifies authenticity with precision.</li>
                    <li><b>Heatmap Analysis:</b> Highlights manipulated regions.</li>
                    <li><b>Real-Time Engine:</b> Processes live feeds instantly.</li>
                </ul>
            </details>
        </div>
        """, unsafe_allow_html=True)

    elif page == "Model Stats":
        st.markdown('<h2 class="title">Model Performance Insights</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if os.path.exists(ACCURACY_PLOT):
                st.image(ACCURACY_PLOT, caption="Accuracy Over Epochs")
            else:
                st.markdown('<div class="card" style="text-align:center;padding:1rem;">Add accuracy plot here</div>', unsafe_allow_html=True)
        with col2:
            if os.path.exists(LOSS_PLOT):
                st.image(LOSS_PLOT, caption="Loss Over Epochs")
            else:
                st.markdown('<div class="card" style="text-align:center;padding:1rem;">Add loss plot here</div>', unsafe_allow_html=True)
        with col3:
            if os.path.exists(CONFUSION_PLOT):
                st.image(CONFUSION_PLOT, caption="Confusion Matrix")
            else:
                st.markdown('<div class="card" style="text-align:center;padding:1rem;">Add confusion matrix here</div>', unsafe_allow_html=True)

    elif page == "Live Demo":
        st.markdown('<h2 class="title">Live Webcam Analysis</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="webcam-card">', unsafe_allow_html=True)
            run = st.checkbox("Start Webcam Detection", key="webcam_toggle")
            
            # Initialize with a black placeholder image
            placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)  # Black image
            FRAME_WINDOW = st.image(placeholder_image, caption="Webcam Feed", use_column_width=True)
            status_placeholder = st.empty()
            confidence_history = deque(maxlen=20)  # Store last 20 confidence scores
            
            if run:
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    st.error("Cannot access webcam. Please check your device.")
                else:
                    while run:
                        ret, frame = camera.read()
                        if not ret:
                            st.error("Failed to capture frame.")
                            break
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result, confidence, _ = predict_image(frame_rgb)
                        
                        FRAME_WINDOW.image(frame_rgb)
                        
                        if result:
                            result_class = "result-real" if result == "Real" else "result-fake"
                            status_placeholder.markdown(f"""
                            <div class="webcam-overlay">
                                <span class="{result_class}">Prediction: {result} ({confidence:.2f}%)</span>
                            </div>
                            """, unsafe_allow_html=True)
                            confidence_history.append(confidence)
                        else:
                            status_placeholder.markdown('<div class="webcam-overlay">Processing...</div>', unsafe_allow_html=True)
                        
                        # Confidence trend plot
                        if len(confidence_history) > 1:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(y=list(confidence_history), mode='lines+markers', 
                                                    line=dict(color='#88c0d0'), marker=dict(size=8)))
                            fig.update_layout(title="Confidence Trend", xaxis_title="Frame", yaxis_title="Confidence (%)",
                                            plot_bgcolor="#1e2a44", paper_bgcolor="#1e2a44", font=dict(color="#d9e1e8"))
                            st.plotly_chart(fig, use_container_width=True)
                        
                        time.sleep(0.05)
                    camera.release()
                    FRAME_WINDOW.image(placeholder_image)
                    status_placeholder.empty()
            st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #5e81ac; font-size: 0.9rem;">
        © 2025 Deepfake Detection & Prevention System | Powered by Pratham Katariya
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
