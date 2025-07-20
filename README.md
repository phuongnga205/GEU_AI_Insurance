# GEU\_AI\_Insurance 🚗🔧

## 👩‍💼 Author Information

- **University**: University of Information Technology – Vietnam National University, Ho Chi Minh City
- **Student ID**: 23520992
- **Full Name**: Le Ngoc Phuong Nga

## 🌟 Project Overview

GEU\_AI\_Insurance is a deep learning-driven AI system designed to detect and segment damaged regions on vehicles from images. Additionally, the system estimates damage costs and provides a web interface for users to upload car images and view prediction results in real time.

## 🔍 Main Features

- Train a segmentation model to identify damaged areas on cars.
- Visualize predicted damage regions overlayed on original images.
- Estimate damage cost based on the proportion of damaged area.
- Offer a simple web interface for image upload and result display (currently under development).

## ⚙️ Technologies Used

### 🧠 Backend - AI & Deep Learning

- **Python 3.8+**
- **PyTorch**: Primary framework for building and training the segmentation model.
  - *Why PyTorch?* Flexible API and strong community support.
- **OpenCV**: Image preprocessing and display utilities.
- **NumPy**: Efficient numerical data manipulation.
- **Matplotlib**: Visualizing segmentation outputs.

### 🌐 Frontend - Web Interface (Upcoming)

- **HTML5/CSS3** with Jinja2 templates: Create a lightweight, responsive UI for image upload.
- **Flask**: Lightweight Python web framework to serve the ML model via API.
  - *Why Flask?* Rapid development of MVPs and easy ML integration.

## 📁 Project Structure

```plaintext
GEU_AI_Insurance/
├─ app/
│  └─ web/                    # Web demo (Flask)
│     ├─ __init__.py          # Python package initializer
│     ├─ main.py              # Flask application entry point
│     ├─ utils.py             # Model loading & image processing functions
│     ├─ templates/           # HTML templates
│     │  └─ index.html        # Upload form & result display
│     └─ static/              # Static assets
│         ├─ uploads/         # Original user-uploaded images
│         └─ results/         # Overlayed output images
├─ sample_car_damage/         # Example car damage images
├─ annotations/               # LabelMe annotation files
├─ source_code/
│  ├─ backend/                # Model training and evaluation scripts
│  └─ frontend/               # Web interface code (Flask)
├─ models/checkpoints/        # Trained model checkpoints (.pth)
├─ screenshots/               # Demo screenshots
├─ train.py                   # Training script for the model
├─ requirements.txt           # Python dependencies
├─ README.md                  # Project description and usage guide
└─ TRAINING.md                # Data processing & training details
```

## 📞 Contact

- **GitHub**: [https://github.com/phuongnga205/GEU\_AI\_Insurance](https://github.com/phuongnga205/GEU_AI_Insurance)
- **Email**: [23520992@gm.uit.edu.vn](mailto:23520992@gm.uit.edu.vn)

---

Thank you for exploring the GEU\_AI\_Insurance project! Feel free to open an issue or submit a pull request on GitHub for feedback and contributions.

