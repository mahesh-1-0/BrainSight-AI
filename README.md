<div align="center">

<img src="assets/logo.png" alt="BrainSight-AI Logo" width="280"/>

# BrainSight-AI

### Interpretable Brain Tumor Detection with Visual Explainability & AI Medical Reporting

[![Live Demo](https://img.shields.io/badge/Live_Demo-Railway-success?style=for-the-badge&logo=railway)](https://brainsight-ai-production.up.railway.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/TensorFlow-DenseNet-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Explainability](https://img.shields.io/badge/Explainable_AI-Grad--CAM-red?style=for-the-badge)](#-visual-explainability-grad-cam)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<p align="center">
  <b>Developed in Collaboration with AICTE & IBM SkillsBuild (CSRBOX)</b>
</p>

</div>

---

## 📸 Product Walkthrough

### 1. Patient Portal & MRI Scan Ingestion
| Landing Interface | Multi-Format MRI Upload & Sample Scans |
| :---: | :---: |
| <img src="assets/homepage.png" width="460" alt="Landing Page"/> | <img src="assets/MRIscan.png" width="460" alt="MRI Scan Upload"/> |

### 2. Explainable Classification & Clinical Reports
| Grad-CAM Visual Heatmap | LLM Clinical Analysis & PDF Export |
| :---: | :---: |
| <img src="assets/MLpage.png" width="460" alt="Grad-CAM Tumor Heatmap"/> | <img src="assets/AIpage.png" width="460" alt="AI Generated Clinical Report"/> |

---

## 📌 Executive Summary

**BrainSight-AI** is a clinical decision-support system designed to accelerate neuro-oncology screening and eliminate black-box opacity in deep learning diagnostics. Traditional diagnostic workflows rely heavily on continuous radiologist availability, creating severe screening bottlenecks in under-resourced medical environments.

BrainSight-AI implements an end-to-end diagnostic pipeline:
1. **DenseNet Deep Feature Extraction:** Accurately classifies brain MRI scans across multi-class cohorts (Glioma, Meningioma, Pituitary, and Normal).
2. **Visual Explainability via Grad-CAM:** Generates localized class activation heatmaps to highlight precise anatomical regions driving model predictions, ensuring clinical transparency.
3. **Automated Generative Medical Reporting:** Uses the Gemini LLM to synthesize radiological indicators into structured preliminary clinical findings, downloadable as an audit-ready PDF.

Aligned directly with **UN SDG 3 (Good Health and Well-Being)**, the project aims to reduce diagnostic turnaround latency and expand accessible early detection tools.

---

## ✨ Core Features

* **DenseNet CNN Backbone:** Harnesses dense feature connectivity and gradient reuse for fine-grained detection of subtle tumor margins.
* **Explainable AI (XAI):** Integrated Gradient-weighted Class Activation Mapping (Grad-CAM) to verify that inferences focus on pathological tissue rather than imaging artifacts.
* **Intelligent Clinical Synthesis:** Automatically produces structured observations including suspected severity, anatomical localization, and recommended clinical follow-ups.
* **Exportable PDF Documentation:** Complete PDF report generation via ReportLab, compiling patient metadata, confidence intervals, and visualization maps for diagnostic review.
* **Cloud-Native Deployment:** Fully containerized with Docker and served live on Railway PaaS.

---

## 🛠️ Architecture & Tech Stack
[ Cranial MRI Input ]
│
├──> DenseNet CNN Classifier ──────> [ Prediction & Confidence Score ]
│
├──> Grad-CAM Layer Visualizer ───> [ Anatomical Activation Heatmap ]
│
└──> Gemini LLM Engine ───────────> [ Structured Clinical Findings ]
│
[ Downloadable PDF Report ]

| Domain | Tools & Technologies |
| :--- | :--- |
| **Core Language** | Python 3.10+ |
| **Deep Learning** | TensorFlow, Keras, DenseNet CNN |
| **Model Interpretability** | Grad-CAM (Gradient-Weighted Class Activation Mapping) |
| **Computer Vision** | OpenCV, NumPy, Matplotlib |
| **Clinical Text Synthesis** | Google Gemini API |
| **Web Server & Backend** | Flask, Gunicorn |
| **Report Generation** | ReportLab (PDF Engine) |
| **Deployment & CI/CD** | Docker, Railway Cloud Platform, Git/GitHub |

---

## 📊 Model Evaluation & Benchmarks

The deep learning architecture was evaluated across standardized brain MRI cohorts:

* **Overall Classification Accuracy:** ~88% – 90% across validated multi-class test sets.
* **Convergence Behavior:** Steady convergence during training, achieving ~96% training accuracy with minimal loss divergence and low overfitting.
* **Balanced Diagnostics:** High recall and precision across Glioma and Pituitary classes, lowering false negative risks during preliminary triage.

---

## 🚀 Local Installation & Setup

### Prerequisites
* Python 3.10 or higher
* Git
* A Google Gemini API Key

### 1. Clone the Repository
```bash
git clone [https://github.com/mahesh-1-0/BrainSight-AI.git](https://github.com/mahesh-1-0/BrainSight-AI.git)
cd BrainSight-AI
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a .env file in the root directory:
```bash
Code snippet
GEMINI_API_KEY=your_actual_gemini_api_key
FLASK_ENV=development
PORT=5000
```

### 5. Run the Application
```bash
python app.py
Open http://127.0.0.1:5000/ in your web browser.
```

