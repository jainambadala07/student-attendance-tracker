"""
Bulk-import student photos and retrain the model.

Usage (from the attendance folder, venv active):
    python import_photos.py photos

Folder layout:  photos/<rollno>_<Name-With-Dashes>/*.jpg|*.jpeg|*.png
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"
MODEL = BASE / "model.yml"
LABELS = BASE / "labels.json"
SIZE = (200, 200)

cascade = cv2.CascadeClassifier(str(BASE / "haarcascade_frontalface_default.xml"))
if cascade.empty():
    sys.exit('Cascade failed to load. Run: python -m pip install "opencv-contrib-python<5"')

if len(sys.argv) != 2 or not Path(sys.argv[1]).is_dir():
    sys.exit("Usage: python import_photos.py <photos_folder>")

# ---- 1. detect, crop and save faces ----
DATASET.mkdir(exist_ok=True)
for student in sorted(p for p in Path(sys.argv[1]).iterdir() if p.is_dir()):
    out = DATASET / student.name
    out.mkdir(exist_ok=True)
    ok = skipped = 0
    for img_path in sorted(student.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        if len(faces) != 1:           # need exactly one face
            skipped += 1
            continue
        x, y, w, h = faces[0]
        face = cv2.resize(gray[y:y + h, x:x + w], SIZE)
        cv2.imwrite(str(out / f"imp_{img_path.stem}.jpg"), face)
        ok += 1
    print(f"{student.name}: {ok} saved, {skipped} skipped")

# ---- 2. train ----
faces, labels, label_map = [], [], {}
for idx, folder in enumerate(sorted(p for p in DATASET.iterdir() if p.is_dir())):
    roll, _, name = folder.name.partition("_")
    label_map[idx] = {"roll": roll, "name": name.replace("-", " ")}
    for f in folder.glob("*.jpg"):
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            faces.append(cv2.resize(im, SIZE))
            labels.append(idx)

if not faces:
    sys.exit("No usable face images found.")
rec = cv2.face.LBPHFaceRecognizer_create()
rec.train(faces, np.array(labels))
rec.write(str(MODEL))
LABELS.write_text(json.dumps(label_map))
print(f"Trained on {len(faces)} images of {len(set(labels))} students.")
