# Brain Tumor MRI Analysis Application

This is a Streamlit-based web application for detecting brain tumors in MRI scans using Google's Gemini AI and image processing techniques. It accepts MRI images as input, processes them, and provides tumor detection results with bounding boxes and confidence scores.

## Features
- Upload MRI scans in `.jpg`, `.jpeg`, or `.png` formats.
- Analyze MRI scans for potential tumors using Google's Gemini AI.
- Enhance MRI image quality for better visualization.
- Display detected tumor regions with bounding boxes and confidence scores.
- View detection details and raw analysis results.

---

## Prerequisites

### Tools Required
- Python (3.8 or higher recommended)
- `pip` for installing dependencies
- Virtual environment tools (e.g., `venv` or `conda`)

---

## Installation Guide

### Step 1: Clone the Repository
```bash
git clone <repository_url>
cd <repository_name>
```

### Step 2: Create a Virtual Environment

#### For Unix/MacOS
```bash
python3 -m venv venv
source venv/bin/activate
```

#### For Windows
```bash
python -m venv venv
venv\Scripts\activate
```

> **Note:** Ensure the virtual environment is activated before proceeding to the next steps.

### Step 3: Install Dependencies
All required dependencies are listed in `requirements.txt`. Install them with the following command:
```bash
pip install -r requirements.txt
```

### Step 4: Set Up API Key
- Add your Google Generative AI API key to Streamlit secrets or as an environment variable.

#### Option 1: Add to Streamlit Secrets
Create a `.streamlit/secrets.toml` file with the following content:
```toml
GENAI_API_KEY = "your_api_key_here"
```

#### Option 2: Set as an Environment Variable
```bash
export GENAI_API_KEY="your_api_key_here"  # For Unix/MacOS
set GENAI_API_KEY="your_api_key_here"  # For Windows
```

---

## Running the Application

To start the Streamlit app, run:
```bash
streamlit run app.py
```

This will launch the application in your default web browser. If it doesn’t open automatically, navigate to the URL shown in the terminal (usually `http://localhost:8501`).

---

## Usage Instructions

1. Upload an MRI scan using the file uploader on the web interface.
2. Click the **Analyze Scan** button to detect potential tumor regions.
3. View the results, including:
   - Enhanced MRI image with tumor bounding boxes.
   - Detection details such as coordinates and confidence levels.
   - Raw analysis text for advanced users.

---

## Requirements
- **Python Packages** (from `requirements.txt`):
  - `streamlit`
  - `Pillow`
  - `google-generativeai`
  - `opencv-python`
  - `numpy`
  - `logging`
