# Multimodal Emotion Recognition

A robust, end-to-end multimodal machine learning application that detects human emotions by analyzing three primary data streams: text (NLP), audio (speech), and video (facial expressions). The systems processes each modality independently via dedicated inference modules and blends them using a fusion network to deliver a highly accurate unified emotion prediction. 

This repository is optimized for containerized deployment and is compatible with Hugging Face Spaces using the Docker SDK.

---

## 🚀 Live Demo  
The live application is hosted on Hugging Face Spaces:  
https://az-is21-multimodal-emotion-recognition.hf.space/

---

## 📁 Repository Structure

```text  
├── fastapi_app/          # FastAPI backend and frontend assets (HTML, CSS, JS)  
├── models/               # Directory for storing trained weights and model configurations  
├── .gitattributes        # Git LFS configuration for tracking large model files  
├── .gitignore            # Specifying intentionally untracked files to ignore  
├── Dockerfile            # Docker configuration for containerized deployment  
├── README.md             # Project documentation  
├── app.py                # Main entry point for the interface/orchestration  
├── audio_inference.py    # Speech emotion recognition processing script  
├── fusion.py             # Multimodal fusion layer (combines NLP, audio, and video predictions)  
├── nlp_inference.py      # Textual emotion recognition processing script  
├── requirements.txt      # Python dependencies  
├── tree.txt              # Text layout of the project tree  
└── video_inference.py    # Facial expression emotion recognition processing script  
```

---

## 🛠️ System Architecture & Workflow

1. **Modality Inferences**:  
   * **NLP (`nlp_inference.py`)**: Evaluates textual tone, semantics, and sentiment analysis.  
   * **Audio (`audio_inference.py`)**: Analyzes acoustic features such as pitch, tone, and energy from speech signals.  
   * **Video (`video_inference.py`)**: Examines facial landmarks, expressions, and micro-movements frame-by-frame.  
2. **Fusion Layer (`fusion.py`)**: Receives the probability matrices or feature embeddings from all three pipelines and computes a unified final prediction using a decision-level or feature-level fusion mechanism.  
3. **Application & Deployment (`app.py` & `fastapi_app`)**: Combines the pipeline into a web service using FastAPI, rendering a fully responsive frontend built with HTML, CSS, and JavaScript.

---

## ⚙️ Installation & Setup

### Prerequisites  
* Python 3.9+ or Docker installed locally.  
* Git LFS initialized (if downloading large model binaries).

### Local Python Setup

1. **Clone the repository:**  
   ```bash  
   git clone https://github.com  
   cd multimodal-emotion-recognition  
   ```

2. **Set up a virtual environment:**  
   ```bash  
   python -m venv venv  
   source venv/bin/activate  # On Windows use: venv\Scripts\activate  
   ```

3. **Install dependencies:**  
   ```bash  
   pip install -r requirements.txt  
   ```

4. **Run the application:**  
   ```bash  
   python app.py  
   ```  
   Navigate to `http://localhost:8000` (or the port specified in your console) to view the application interface.

---

## 🐳 Docker Deployment

The application is fully containerized using a `Dockerfile`. This setup is required for hosting on Hugging Face Spaces via the Docker SDK.

### Build the Image locally:  
```bash  
docker build -t multimodal-emotion-recognition .  
```

### Run the Container:  
```bash  
docker run -p 7860:7860 multimodal-emotion-recognition  
```  
Open your browser and navigate to `http://localhost:7860`.

---

## 🔧 Core Components and Dependencies

The solution leverages modern AI and web frameworks including:  
* **FastAPI**: For high-performance async API processing.  
* **Deep Learning Frameworks**: Libraries specified in `requirements.txt` (such as PyTorch, TensorFlow, or Transformers) to execute state-of-the-art vision, audio, and text modeling.  
* **Web UI (HTML/CSS/JS)**: Clean interface enabling real-time file upload or web-cam capture streams to evaluate performance dynamically.

---

## 📝 License  
This project is open-source. Please check the repository settings or LICENSE file for exact permission mappings.
