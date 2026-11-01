import json
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Config ----------
DATASET = Path("dataset")
DATASET.mkdir(exist_ok=True)
MODEL_PATH = Path("model.yml")
LABELS_PATH = Path("labels.json")
DB_PATH = "attendance.db"
FACE_SIZE = (200, 200)
THRESHOLD = 70  # LBPH distance: LOWER = better match. Tune this (60-80).

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------- Database ----------
def get_db():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll TEXT NOT NULL,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            UNIQUE(roll, date)
        )"""
    )
    return con


def mark_attendance(roll, name):
    """Returns True if newly marked, False if already marked today."""
    now = datetime.now()
    con = get_db()
    try:
        con.execute(
            "INSERT INTO attendance (roll, name, date, time) VALUES (?,?,?,?)",
            (roll, name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")),
        )
        con.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        con.close()


# ---------- Vision helpers ----------
def decode(uploaded):
    arr = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def detect_faces(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = CASCADE.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )
    return gray, faces


def crop(gray, box):
    x, y, w, h = box
    return cv2.resize(gray[y : y + h, x : x + w], FACE_SIZE)


def train_model():
    faces, labels, label_map = [], [], {}
    folders = sorted(p for p in DATASET.iterdir() if p.is_dir())
    for idx, folder in enumerate(folders):
        roll, _, name = folder.name.partition("_")
        label_map[idx] = {"roll": roll, "name": name.replace("-", " ")}
        for img_path in folder.glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(cv2.resize(img, FACE_SIZE))
                labels.append(idx)
    if len(set(labels)) < 1 or not faces:
        return 0, 0
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.write(str(MODEL_PATH))
    LABELS_PATH.write_text(json.dumps(label_map))
    return len(label_map), len(faces)


# ---------- Pages ----------
def page_register():
    st.header("Register a student")
    c1, c2 = st.columns(2)
    roll = c1.text_input("Roll number").strip()
    name = c2.text_input("Full name").strip()

    if "cam_n" not in st.session_state:
        st.session_state.cam_n = 0

    if not roll or not name:
        st.info("Enter roll number and name, then take 10-15 photos "
                "(vary angle, lighting, expression).")
        return

    folder = DATASET / f"{roll}_{name.replace(' ', '-')}"
    saved = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
    st.write(f"Photos saved for this student: **{saved}**")

    shot = st.camera_input("Take photo", key=f"cam_{st.session_state.cam_n}")
    if shot:
        bgr = decode(shot)
        gray, faces = detect_faces(bgr)
        if len(faces) != 1:
            st.warning(f"Found {len(faces)} faces. Need exactly 1. Retake.")
        else:
            folder.mkdir(exist_ok=True)
            face = crop(gray, faces[0])
            fname = folder / f"{datetime.now().strftime('%H%M%S%f')}.jpg"
            cv2.imwrite(str(fname), face)
            st.session_state.cam_n += 1
            st.rerun()

    if st.button("Train / update model"):
        n_students, n_imgs = train_model()
        if n_imgs:
            st.success(f"Trained on {n_imgs} images of {n_students} students.")
        else:
            st.error("No training images found.")


def page_mark():
    st.header("Mark attendance")
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        st.warning("No trained model yet. Register students and train first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    labels = json.loads(LABELS_PATH.read_text())

    shot = st.camera_input("Look at the camera and take a photo")
    if not shot:
        return

    bgr = decode(shot)
    gray, faces = detect_faces(bgr)
    if len(faces) == 0:
        st.warning("No face detected. Try better lighting.")
        return

    for box in faces:
        x, y, w, h = box
        label, dist = recognizer.predict(crop(gray, box))
        if dist < THRESHOLD:
            s = labels[str(label)]
            is_new = mark_attendance(s["roll"], s["name"])
            text = f"{s['name']} ({dist:.0f})"
            colour = (0, 200, 0)
            if is_new:
                st.success(f"Marked present: {s['name']} ({s['roll']})")
            else:
                st.info(f"{s['name']} already marked today.")
        else:
            text, colour = "Unknown", (0, 0, 255)
            st.error("Face not recognised.")
        cv2.rectangle(bgr, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(bgr, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    st.image(bgr, channels="BGR")


def page_dashboard():
    st.header("Attendance records")
    con = get_db()
    df = pd.read_sql("SELECT roll, name, date, time FROM attendance "
                     "ORDER BY date DESC, time DESC", con)
    con.close()
    if df.empty:
        st.info("No records yet.")
        return

    c1, c2 = st.columns(2)
    dates = c1.multiselect("Date", sorted(df["date"].unique(), reverse=True))
    students = c2.multiselect("Student", sorted(df["name"].unique()))
    if dates:
        df = df[df["date"].isin(dates)]
    if students:
        df = df[df["name"].isin(students)]

    st.dataframe(df, use_container_width=True)
    st.metric("Records shown", len(df))
    st.download_button("Download CSV", df.to_csv(index=False),
                       "attendance.csv", "text/csv")


# ---------- Main ----------
st.set_page_config(page_title="Attendance Tracker", page_icon="📋")
st.title("Student Attendance Tracker")
page = st.sidebar.radio("Menu", ["Mark attendance", "Register student", "Dashboard"])
{"Mark attendance": page_mark,
 "Register student": page_register,
 "Dashboard": page_dashboard}[page]()