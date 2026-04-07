

# 🚧 AI-Powered Industrial Machine Utilization & Activity Monitoring Dashboard

A real-time computer vision system designed to monitor construction machinery, analyze operational efficiency, and classify activities using deep learning.

This project integrates **YOLOv5**, **LSTM**, and a **Streamlit dashboard** to provide actionable insights into machine usage (Active vs Idle) and task-level activities (e.g., Digging, Loading).

---

## 📌 Key Features

* 🎯 **Object Detection** using YOLOv5
* 🔄 **Activity Recognition** using LSTM (sequence-based prediction)
* ⚙️ **Operational Status Detection** (Active vs Idle)
* 📊 **Interactive Dashboard** built with Streamlit
* 🗄️ **PostgreSQL Integration** for structured logging
* 📁 **Automated JSON Export** for session data
* 🎥 **Video Processing Support** (not limited to images)

---

## 🧠 Model Overview

### 1. Object Detection (YOLOv5)

Trained on a custom dataset from Roboflow with the following classes:

* **0** → Closed White Cars
* **1** → Dump Truck
* **2** → Excavator

---

### 2. Activity Recognition (LSTM)

A sequence-based deep learning model trained to classify machinery actions:

* Digging
* Loading
* Dumping
* Swinging

---

### 3. Motion Detection

Custom logic to determine:

* **Active حالت تشغيل**
* **Idle حالة خمول**

---

## 📂 Project Structure & File Descriptions

### 1. `roboflow.py`

* Used for training the YOLOv5 model
* Dataset sourced from Roboflow
* Handles data preprocessing and training pipeline

---

### 2. `testing1.py`

* Evaluates the pretrained YOLOv5 model
* Runs detection on **videos instead of images**
* Used to validate real-world performance

---

### 3. `testing2.py`

* Detects whether objects are:

  * **Static (Idle)**
  * **Moving (Active)**
* Based on motion tracking logic

---

### 4. `savingposes.py`

* Responsible for dataset creation for LSTM
* Extracts frames from YouTube videos (every 30 seconds)
* Manually labels activities:

  * Digging
  * Loading
  * Dumping
  * Swinging
* Trains and saves the model as `.pth` file

---

### 5. `database.py`

* Creates and manages PostgreSQL tables
* Stores:

  * Machine activity logs
  * Status (Idle/Active)
  * Timestamps

---

### 6. `app.py`

* Main application file
* Integrates:

  * YOLOv5 model
  * LSTM activity model (`.pth`)
  * Streamlit dashboard
* Handles:

  * Real-time video processing
  * Database logging
  * JSON session export

---

## 🖥️ User Interface

The Streamlit dashboard provides:

* 📤 **Upload Video Button**
* ⏹️ **Stop Processing Button**
* 💾 **Save Session as JSON**
* ⏱️ **Live Tracking of:**

  * Total Active Time
  * Total Idle Time

---

## 🔄 Workflow

1. Upload a video through the interface
2. YOLOv5 detects machinery in each frame
3. Motion analysis determines Active/Idle state
4. LSTM model predicts activity type
5. Results are:

   * Displayed in real-time
   * Stored in PostgreSQL
   * Exported as JSON

---

## 🛠️ Technologies Used

* Python
* YOLOv5
* PyTorch
* OpenCV
* Streamlit
* PostgreSQL


