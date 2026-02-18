# Debris Intelligence

AI-powered underwater debris detection and analytics platform.

## Features
- YOLOv8-based debris detection
- Image enhancement (CLAHE)
- Video tracking + heatmap generation
- FastAPI backend
- Modern frontend dashboard

## Tech Stack
- Python
- FastAPI
- YOLOv8
- OpenCV
- HTML / CSS / JavaScript

## How to Run

### Backend
cd backend  
uvicorn main:app --reload  

### Frontend
cd frontend  
python -m http.server 5500  

Then open:
http://127.0.0.1:5500
