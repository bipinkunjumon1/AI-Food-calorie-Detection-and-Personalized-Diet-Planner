# 🍎 AI Powered Food Calorie Detection with Personalized Diet Recommendation

## 📌 Overview

AI Powered Food Calorie Detection with Personalized Diet Recommendation is an intelligent healthcare and nutrition management system that uses Artificial Intelligence, Deep Learning, and Machine Learning to detect food items from images, estimate calories, and generate personalized diet plans.

The system combines:
- YOLOv8 Object Detection
- YOLOv8 Classification
- Decision Tree Recommendation System
- Flask Web Application
- SQLite Database

to provide real-time food recognition and nutritional analysis.

---

# 🚀 Features

✅ Food Image Recognition using YOLOv8  
✅ Indian Food Detection using Object Detection  
✅ Western Food Classification  
✅ Automatic Calorie Estimation  
✅ Personalized Diet Recommendation  
✅ Nutrition Tracking System  
✅ Daily Intake Monitoring  
✅ User Authentication System  
✅ Real-Time Progress Tracking  
✅ SQLite Database Integration  

---

# 🧠 AI Models Used

| Model | Purpose |
|---|---|
| YOLOv8 Detection Model | Detect Indian food items |
| YOLOv8 Classification Model | Classify Western foods |
| Decision Tree Classifier | Personalized diet recommendation |

---

# 🛠️ Technologies Used

## Frontend
- HTML
- CSS
- Bootstrap
- JavaScript

## Backend
- Python
- Flask

## AI / ML
- YOLOv8
- OpenCV
- Scikit-learn
- NumPy
- Pandas

## Database
- SQLite

---

# 📂 Project Structure

```bash
AI-Powered-Food-Calorie-Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── dataset/
├── models/
├── static/
├── templates/
├── uploads/
├── best.pt
├── diet_model.pkl
└── database/
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/AI-Powered-Food-Calorie-Detection.git
```

---

## 2️⃣ Navigate to Project Folder

```bash
cd AI-Powered-Food-Calorie-Detection
```

---

## 3️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

## 4️⃣ Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

## 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

# 🧾 Workflow

1. User uploads food image  
2. YOLOv8 detects food items  
3. Calories are estimated  
4. Nutrition values are calculated  
5. Decision Tree generates personalized diet plan  
6. Daily intake is tracked  
7. Progress dashboard displays reports  

---

# 🖼️ Dataset

The project uses:
- Food image datasets
- Labelled YOLO annotation files
- Diet recommendation datasets

## Food Dataset Includes
- Indian food images
- Western food images
- Bounding boxes
- Class labels
- Nutritional mappings

---

# 🧪 Model Training

## YOLOv8 Training

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640
)
```

---

# 📊 Evaluation Metrics

- Precision
- Recall
- mAP (Mean Average Precision)
- Accuracy Score

---

# 🎯 Future Scope

- Mobile App Integration
- Real-Time Video Detection
- Cloud Deployment
- Better Portion Size Estimation
- AI Nutrition Chatbot
- Wearable Device Support

---

# ⚠️ Limitations

- Accuracy depends on image quality
- Portion size estimation is approximate
- Limited dataset size
- API dependency for nutrition data

---

# 👨‍💻 Authors

- Bipin Kunjumon
- Final Year B.Tech IT Project

---

# 📜 License

This project is developed for educational and academic purposes.

---

# ⭐ Conclusion

This project demonstrates the integration of Artificial Intelligence, Computer Vision, and Machine Learning for smart nutrition analysis and personalized healthcare recommendations.
