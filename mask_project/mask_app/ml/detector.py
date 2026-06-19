import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import os

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mask_detector.pth")
PROTOTXT_PATH = os.path.join(BASE_DIR, "face_detector", "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(BASE_DIR, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")

# ── device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── load mask classifier once at startup ─────────────────────────────────────
mask_model = models.mobilenet_v2(weights=None)
mask_model.classifier[1] = torch.nn.Linear(mask_model.last_channel, 2)
mask_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
mask_model.to(device)
mask_model.eval()

# ── load face detector once at startup ───────────────────────────────────────
face_net = cv2.dnn.readNet(PROTOTXT_PATH, CAFFEMODEL_PATH)

# ── image transform ───────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def detect_from_frame(frame):
    """
    Takes a BGR numpy frame (from OpenCV / decoded JPEG).
    Returns a list of detections, one per detected face:
        [
            {
                "mask_detected": bool,
                "confidence":    float,   # confidence of the winning class
                "face_crop":     np.ndarray (BGR),  # cropped face region
                "box":           (sx, sy, ex, ey)
            },
            ...
        ]
    Returns an empty list if no face is found.
    """
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        sx, sy, ex, ey = box.astype(int)

        sx, sy = max(0, sx), max(0, sy)
        ex, ey = min(w - 1, ex), min(h - 1, ey)

        face_crop = frame[sy:ey, sx:ex]
        if face_crop.size == 0:
            continue

        # convert BGR → RGB for PIL / torchvision
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        img_tensor = transform(Image.fromarray(face_rgb)).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.softmax(mask_model(img_tensor), dim=1)[0]

        mask_prob    = pred[0].item()
        no_mask_prob = pred[1].item()

        mask_detected = mask_prob > no_mask_prob
        confidence    = mask_prob if mask_detected else no_mask_prob

        results.append({
            "mask_detected": mask_detected,
            "confidence":    round(confidence, 4),
            "face_crop":     face_crop,        # BGR numpy array
            "box":           (sx, sy, ex, ey),
        })

    return results
