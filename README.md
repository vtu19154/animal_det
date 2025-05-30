# 🐾 Real-Time Animal Species Detection with YOLO 🚀

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-RealTime-green?logo=opencv)
![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?logo=ultralytics)

Welcome to the **Real-Time Animal Detection System**! This project uses cutting-edge computer vision and deep learning (YOLOv3 & YOLOv8) to identify and localize various animal species in **images, videos, and real-time webcam feeds**. 🧠🎯

---

## 📸 Features

- 🔍 Detects 20+ animal classes in real-time
- 🧠 Two models:
  - `YOLOv3` for lightweight detection using OpenCV
  - `YOLOv8` with a trained custom model via Ultralytics
- 🖼️ Upload and analyze images
- 🎥 Upload and analyze videos
- 🎦 Live webcam animal detection
- 📊 Displays bounding boxes, class names, and confidence
- 📝 Streamlit-powered GUI with interactive sidebar

---

## 🦁 Detected Animal Classes

- 🐶 Dog, 🐱 Cat, 🐘 Elephant, 🦓 Zebra, 🐅 Tiger, 🦁 Lion, 🦊 Fox, 🐼 Panda, 🐴 Horse, 🐮 Cow, and many more...

---

## 🧠 Models Used

| Model      | Framework    | Use Case         |
|------------|--------------|------------------|
| YOLOv3     | OpenCV DNN   | Lightweight real-time detection |
| YOLOv8     | Ultralytics  | Custom trained model for improved accuracy |

---

## 🛠️ How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/vtu19154/animal_det.git
cd animal_det


Here's a **professional and engaging `README.md`** for your Animal Detection project using YOLO, with **emojis**, **badges**, and **sections** that are suitable for GitHub.

---


### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

> 📁 Ensure the following files are in the correct paths:
>
> * `yolov3.weights`, `yolov3.cfg`, `coco.names` for the OpenCV model
> * `best.pt` for the YOLOv8 custom model (place under `runs/detect/train/weights/`)

---

## 🧪 Sample Usage

* Press `Start Webcam` to begin real-time detection 🖥️
* Upload `.jpg`, `.png`, or `.mp4` files for offline inference
* Watch predictions rendered directly in the web app 🎯

---

## 📁 Project Structure

```
animal_det/
├── app.py                    # Streamlit app using YOLOv8
├── yolov3_inference.py       # OpenCV real-time webcam detection
├── yolov3.weights, .cfg      # Pretrained YOLOv3 files
├── coco.names                # COCO class labels
├── runs/                     # YOLOv8 trained weights
├── logs/                     # Log files
└── requirements.txt          # Python dependencies
```

---

## 🤖 Requirements

* Python 3.8+
* OpenCV
* Streamlit
* Ultralytics (YOLOv8)
* Pillow
* NumPy

Install everything with:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

* For webcam detection, ensure your system has a functional camera.
* For YOLOv8, the `best.pt` model must be trained and present in the correct directory.
* Modify `MODEL_DIR` in `app.py` if your weights are elsewhere.

---

## 🙌 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [YOLOv3 Weights](https://pjreddie.com/darknet/yolo/)
* [Streamlit](https://streamlit.io/)

---

## 📬 Contact

If you like this project, feel free to ⭐ it!

For questions or collaboration:
📧 [vtu19154@example.com](mailto:vtu19154@example.com)
👨‍💻 [Your GitHub Profile](https://github.com/vtu19154)

---

> 🚀 *Empowering wildlife conservation with AI and vision!*

```

---

Let me know if you'd like me to tailor the README for **deployment**, **Docker**, or **cloud hosting** (e.g., Streamlit Cloud, Heroku).
```

