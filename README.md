# 🚗 AI-Powered Vehicle Condition Assessment

## Introduction

This project is an **AI-powered vehicle condition assessment system** designed for rental businesses to automate and simplify vehicle inspections. The system uses computer vision to detect damages by comparing vehicle images taken at pickup and return times.

### What This Project Does

- **Automated Damage Detection**: Uses YOLO AI model to detect vehicle damages
- **Before/After Comparison**: Compares pickup and return images to identify new damages  
- **Cost Estimation**: Provides estimated repair costs based on detected damages
- **Visual Reporting**: Displays comprehensive damage assessment reports
- **REST API**: Exposes endpoints for integration with other systems

## Real Application Results

### Actual Damage Assessment Report
![Real Damage Assessment Results](./data/output/Damage-Assessment-Result.png)

*This screenshot shows the actual results from our AI system:*
- **1 New Damage** detected
- **$444.48** estimated repair cost  
- **1.3/10** severity score
- **Side-by-side comparison** of pickup vs return images
- **Damage count**: 1 pre-existing, 2 new damages detected

## Technology Stack

- **Backend**: FastAPI (Python)
- **AI Model**: YOLOv8 (Ultralytics) 
- **Frontend**: Streamlit
- **Computer Vision**: OpenCV
- **Image Processing**: Pillow, NumPy

## Quick Start

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/vehicle_damage_assessment.git
cd vehicle-damage-assessment
```

### Step 2: Set Up Virtual Environment
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Requirements
```bash
pip install -r requirements.txt
```

## How to Run the Application

### Run Both Services

**Terminal 1 - Start Backend:**
```bash
python main.py
```
✅ Backend running on: `http://localhost:8000`

**Terminal 2 - Start Frontend:**
```bash
streamlit run streamlit_app.py
```
✅ Frontend running on: `http://localhost:8501`

## How to Use
1. **Upload pickup images** (vehicle when picked up)
2. **Upload return images** (vehicle when returned) 
3. **Click "Assess Damage"** to analyze
4. **View results** similar to the screenshot above
## Features

- ✅ AI-powered damage detection
- ✅ Before/after image comparison  
- ✅ Repair cost estimation
- ✅ Severity assessment
- ✅ REST API endpoints
- ✅ Web interface

## 📊 Project Structure

```
vehicle-damage-assessment/
├── models/
│   └── best.pt              # Pre-trained YOLO model
├── main.py                  # FastAPI backend
├── streamlit_app.py         # Streamlit frontend
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Troubleshooting

**Common Issues:**
- Ensure Python 3.8+ is installed
- Check all requirements are installed: `pip install -r requirements.txt`
- Verify `models/best.pt` exists in project
- Make sure ports 8000 and 8501 are available

## License

This project is developed for the Aspire Software Hiring Sprint 2025.

---

**🎉 Your vehicle damage assessment system is ready!**

Run the application and upload vehicle images to get AI-powered damage analysis like the results shown above! 🚗✨
