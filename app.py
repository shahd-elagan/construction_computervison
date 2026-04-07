import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import psycopg2
import time
import pathlib
import platform
import tempfile
import json
from datetime import datetime
from collections import deque

# 1. Windows Compatibility Fix
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

# --- Streamlit Page Config ---
st.set_page_config(page_title="Machine Utilization UI", layout="wide")
st.title("🏗️ Smart Machine Monitor & Utilization Dashboard")

# --- Initialize Session State ---
if 'stats' not in st.session_state:
    st.session_state.stats = {} 
if 'logs' not in st.session_state:
    st.session_state.logs = [] 

# -----------------------------------------------------------------------------
# 2. Database & JSON Export Logic
# -----------------------------------------------------------------------------
DB_CONFIG = {
    "dbname": "task",
    "user": "postgres",
    "password": "1234", 
    "host": "localhost", 
    "port": "5432"
}

def stream_to_db(label, class_id, status, mode, dwell_time):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        query = """INSERT INTO excavator_logs (object_name, class_id, status, action_mode, dwell_time) 
                   VALUES (%s, %s, %s, %s, %s)"""
        cur.execute(query, (label, class_id, status, mode, round(dwell_time, 2)))
        conn.commit()
        cur.close()
        conn.close()
    except:
        pass

def generate_cv_payload(label, class_id, status, mode, dwell, box, stats, fps):
    active_f = stats.get(class_id, {}).get("active_f", 0)
    idle_f = stats.get(class_id, {}).get("idle_f", 0)
    total_f = active_f + idle_f
    util_per = (active_f / total_f * 100) if total_f > 0 else 0

    return {
        "timestamp": datetime.now().isoformat() + "Z",
        "machine_info": {"machine_id": f"{label.lower()}_01", "class_name": label, "class_id": class_id},
        "state_analysis": {"status": status, "action_mode": mode},
        "time_analysis": {
            "dwell_time_seconds": round(dwell, 2),
            "total_active_seconds": round(active_f / fps, 2),
            "total_idle_seconds": round(idle_f / fps, 2),
            "utilization_percentage": round(util_per, 2)
        },
        "location": {"bbox": [int(x) for x in box]}
    }

# -----------------------------------------------------------------------------
# 3. Models Loading
# -----------------------------------------------------------------------------
class ActionLSTM(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_layers=3, num_classes=4):
        super(ActionLSTM, self).__init__()
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128) 
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc2(torch.relu(self.fc1(out[:, -1, :])))

@st.cache_resource
def init_models():
    yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/AMIT/Desktop/cvvvv/best.pt')
    yolo.conf = 0.5
    lstm = ActionLSTM()
    lstm.load_state_dict(torch.load('C:/Users/AMIT/Desktop/cvvvv/excavator_activities_deep.pth', map_location='cpu'))
    lstm.eval()
    return yolo, lstm

yolo_model, action_model = init_models()
ACTIONS = ['DIGGING', 'LOADING', 'SWINGING', 'DUMPING']

# -----------------------------------------------------------------------------
# 4. Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.header("📂 Controls")
uploaded_file = st.sidebar.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov'])

if st.sidebar.button("💾 Save Session to JSON"):
    if st.session_state.logs:
        with open("final_session_log.json", "w") as f:
            json.dump(st.session_state.logs, f, indent=4)
        st.sidebar.success("Saved to final_session_log.json!")
    else:
        st.sidebar.warning("No data yet.")

if st.sidebar.button("Clear History"):
    st.session_state.stats = {}
    st.session_state.logs = []
    st.rerun()

metrics_area = st.sidebar.empty()
video_area = st.empty()

# -----------------------------------------------------------------------------
# 5. Main Processing Loop
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    registry = {}
    BUFFER_SIZE, SENSITIVITY, IDLE_WAIT_TIME = 15, 10, 1.0
    last_db_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (850, 500))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        df = yolo_model(rgb_frame).pandas().xyxy[0]
        
        current_frame_data = []

        for _, row in df.iterrows():
            label, class_id = row['name'], int(row['class'])
            box = np.array([int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])])
            
            # Simple Tracking Logic
            match_key = None
            curr_center = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])
            for k, v in registry.items():
                if label in k:
                    old_c = np.array([(v['history'][-1][0]+v['history'][-1][2])/2, (v['history'][-1][1]+v['history'][-1][3])/2])
                    if np.linalg.norm(curr_center - old_c) < 80: 
                        match_key = k
                        break
            
            if match_key is None:
                match_key = f"{label}_{time.time()}"
                registry[match_key] = {'history': deque(maxlen=BUFFER_SIZE), 'motion_buf': deque(maxlen=BUFFER_SIZE), 'no_motion_start': None, 'dwell_start': None}
            
            obj = registry[match_key]
            
            # Status Logic
            is_moving = False
            if len(obj['history']) > 0:
                obj['motion_buf'].append((box - obj['history'][-1]).tolist())
                if np.max(np.abs(box - obj['history'][0])) > SENSITIVITY: is_moving = True

            dwell = 0
            if is_moving:
                obj['no_motion_start'], obj['dwell_start'], status, color = None, None, "ACTIVE", (0, 255, 0)
            else:
                if obj['no_motion_start'] is None: obj['no_motion_start'] = time.time()
                status = "IDLE" if (time.time() - obj['no_motion_start'] > IDLE_WAIT_TIME) else "ACTIVE"
                color = (255, 0, 0) if status == "IDLE" else (0, 255, 0)
                if status == "IDLE":
                    if obj['dwell_start'] is None: obj['dwell_start'] = time.time()
                    dwell = time.time() - obj['dwell_start']

            # Update Stats
            if class_id not in st.session_state.stats:
                st.session_state.stats[class_id] = {"active_f": 0, "idle_f": 0, "name": label}
            if status == "ACTIVE": st.session_state.stats[class_id]["active_f"] += 1
            else: st.session_state.stats[class_id]["idle_f"] += 1
            
            obj['history'].append(box)

            # Detect Activity Mode
            mode = "WORKING" if status == "ACTIVE" else "NONE"
            if status == "ACTIVE" and class_id == 2 and len(obj['motion_buf']) == BUFFER_SIZE:
                padded = np.zeros((1, BUFFER_SIZE, 512), dtype=np.float32)
                padded[0, :, :4] = np.array(obj['motion_buf'])
                with torch.no_grad():
                    mode = ACTIONS[torch.argmax(action_model(torch.tensor(padded))).item()].upper()

            # Store Logs for JSON
            payload = generate_cv_payload(label, class_id, status, mode, dwell, box, st.session_state.stats, fps)
            st.session_state.logs.append(payload)
            current_frame_data.append((label, class_id, status, mode, dwell))
            
            # --- Visual Labels ---
            cv2.rectangle(rgb_frame, (box[0], box[1]), (box[2], box[3]), color, 3)
            status_text = f"{label}: {status}"
            if status == "ACTIVE" and class_id == 2: status_text += f" ({mode})"
            if status == "IDLE": status_text += f" {int(dwell)}s"
            
            # Draw text background
            cv2.rectangle(rgb_frame, (box[0], box[1]-25), (box[0]+280, box[1]), color, -1)
            cv2.putText(rgb_frame, status_text, (box[0]+5, box[1]-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Update Live View
        video_area.image(rgb_frame, channels="RGB")

        # Sidebar Update
        with metrics_area.container():
            st.markdown("### 📊 Live Stats")
            for cid, s in st.session_state.stats.items():
                act, idl = s["active_f"]/fps, s["idle_f"]/fps
                st.write(f"**{s['name']}**")
                st.caption(f"Active: {act:.1f}s | Idle: {idl:.1f}s")

        # Database Sync Every 1s
        if time.time() - last_db_time >= 1.0:
            for item in current_frame_data:
                stream_to_db(item[0], item[1], item[2], item[3], item[4])
            last_db_time = time.time()

    cap.release()
else:
    st.info("Upload video to start.")