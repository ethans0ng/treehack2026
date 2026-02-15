import cv2
import gc
from collections import deque
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image
from nanoowl.owl_predictor import OwlPredictor


FRAME_SIZE = (640, 480)
REQUIRED_FRAMES = 120
DETECTION_THRESHOLD = 0.15
NOSE_REFRESH_EVERY = 20
FACE_REFRESH_EVERY = 5
FACE_DETECTION_THRESHOLD = 0.08
FACE_MISS_RELOCK_LIMIT = 15
RELOCK_EYE_MISSES = 12
MAX_PUPIL_HOLD_FRAMES = 4
NOSE_MISS_RELOCK_LIMIT = 25
HEAD_WARNING_PRINT_STRIDE = 4

HEAD_MOVE_THRESHOLD_NORM = 0.06
HEAD_MOVE_EVAL_WINDOW = 10
FRAME_RATE_HZ = 30.0
FRAME_DT_SEC = 1.0 / FRAME_RATE_HZ
SMOOTH_MEDIAN_WINDOW = 5
SMOOTH_MEAN_WINDOW = 3
MIN_VALID_RATIO = 0.55
NYSTAGMUS_CONF_LOW = 40.0
NYSTAGMUS_CONF_HIGH = 70.0
METRIC_BINARY_THRESHOLD = 40.0
HF_BAND_HZ = (0.5, 6.0)
HGN_API_URL = os.getenv("HGN_API_URL", "http://127.0.0.1:8000")
HGN_API_TIMEOUT_SEC = float(os.getenv("HGN_API_TIMEOUT_SEC", "2.5"))
HGN_API_ENABLED = os.getenv("HGN_API_ENABLED", "1") not in {"0", "false", "False", "FALSE", "no", "No", "NO"}
HGN_SUBJECT_NAME = os.getenv("HGN_SUBJECT_NAME", "Treehacks Demo")
HGN_STOP_TIME = os.getenv("HGN_STOP_TIME", "9:00am")
HGN_ARREST_TIME = os.getenv("HGN_ARREST_TIME", "9:30am")
HGN_PUBLISH_ON_SUCCESS = os.getenv("HGN_PUBLISH_ON_SUCCESS", "1") not in {"0", "false", "False", "FALSE", "no", "No", "NO"}


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def clamp_box(box, w, h, pad=0.0):
    x1, y1, x2, y2 = box
    dx = int((x2 - x1) * pad / 2)
    dy = int((y2 - y1) * pad / 2)
    x1 = max(0, x1 - dx)
    x2 = min(w, x2 + dx)
    y1 = max(0, y1 - dy)
    y2 = min(h, y2 + dy)
    return (x1, y1, x2, y2)


def to_writable_rgb_pil(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.ascontiguousarray(rgb))


def detect_boxes(predictor, pil_image, text, threshold=DETECTION_THRESHOLD):
    output = predictor.predict(
        image=pil_image,
        text=[text],
        text_encodings=None,
        threshold=threshold,
    )
    boxes = (
        output.boxes.detach().cpu().numpy().astype(int)
        if hasattr(output, "boxes")
        else np.empty((0, 4), dtype=int)
    )
    scores = (
        output.scores.detach().cpu().numpy()
        if hasattr(output, "scores")
        else np.ones((boxes.shape[0],), dtype=float)
    )
    return boxes, scores


def detect_eye_rois(predictor, image_pil, frame_size=FRAME_SIZE, threshold=DETECTION_THRESHOLD):
    eye_boxes, eye_scores = detect_boxes(predictor, image_pil, "human eye", threshold=threshold)
    if len(eye_boxes) < 2:
        return []
    eye_order = np.argsort(eye_scores)[::-1]
    candidates = [eye_boxes[i] for i in eye_order[:8]]
    candidates = sorted(candidates, key=box_area, reverse=True)[:2]
    return [
        clamp_box(tuple(b), frame_size[0], frame_size[1], 0.25)
        for b in sorted(candidates, key=lambda b: b[0])
    ]


def estimate_nose_from_eyes(eye_rois):
    if len(eye_rois) != 2:
        return None
    eye_l, eye_r = eye_rois
    nx = (eye_l[0] + eye_l[2] + eye_r[0] + eye_r[2]) / 4.0
    ny = (eye_l[1] + eye_l[3] + eye_r[1] + eye_r[3]) / 4.0
    return (int(nx - 8), int(ny + 10), int(nx + 8), int(ny + 24))


def detect_face_centroid(predictor, image_pil, frame_size=FRAME_SIZE, threshold=FACE_DETECTION_THRESHOLD):
    face_boxes, _ = detect_boxes(
        predictor, image_pil, "face", threshold=max(FACE_DETECTION_THRESHOLD, threshold)
    )
    if len(face_boxes) == 0:
        return None
    idx = int(np.argmax([box_area(b) for b in face_boxes]))
    x1, y1, x2, y2 = clamp_box(
        tuple(face_boxes[idx]), frame_size[0], frame_size[1], 0.05
    )
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def get_pupil_location(eye_img_bgr, is_left_eye):
    if eye_img_bgr.size == 0:
        return None

    h, w, _ = eye_img_bgr.shape
    if is_left_eye:
        x_start, x_end = int(w * 0.35), int(w * 0.85)
    else:
        x_start, x_end = int(w * 0.15), int(w * 0.65)

    y_start, y_end = int(h * 0.30), int(h * 0.70)
    crop = eye_img_bgr[y_start:y_end, x_start:x_end]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    mean_val = np.mean(gray)
    _, thresh = cv2.threshold(gray, mean_val - 15, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    crop_area = crop.shape[0] * crop.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10 or area > (crop_area * 0.5):
            continue
        moments = cv2.moments(cnt)
        if moments["m00"] != 0:
            px = int(moments["m10"] / moments["m00"])
            py = int(moments["m01"] / moments["m00"])
            return (px + x_start, py + y_start)
    return None


def _fill_missing(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n == 0:
        return arr
    if np.isfinite(arr).sum() == 0:
        return np.zeros(n, dtype=float)
    x = np.arange(n)
    valid = np.isfinite(arr)
    filled = arr.copy()
    filled[~valid] = np.interp(x[~valid], x[valid], arr[valid])
    return filled


def _rolling_median(arr, window):
    if window <= 1:
        return np.asarray(arr, dtype=float)
    arr = np.asarray(arr, dtype=float)
    half = window // 2
    n = len(arr)
    out = np.empty_like(arr, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(arr[lo:hi])
    return out


def _rolling_mean(arr, window):
    if window <= 1:
        return np.asarray(arr, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _reversal_rate(v):
    v = np.asarray(v, dtype=float)
    active = np.abs(v) > 1e-6
    if active.sum() < 2:
        return 0.0
    signs = np.sign(v[active])
    if len(signs) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(signs)) > 0))


def _band_power_ratio(trace, dt=FRAME_DT_SEC, band=HF_BAND_HZ):
    trace = np.asarray(trace, dtype=float)
    if len(trace) < 8:
        return 0.0
    x = trace - np.mean(trace)
    spectrum = np.fft.rfft(x)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(len(trace), d=dt)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(band_mask):
        return 0.0
    return float(np.sum(power[band_mask]) / max(np.sum(power), 1e-12))


def _lin_norm(v, lo, hi):
    if hi <= lo:
        return 0.0
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def analyze_trace(x_hist, y_hist):
    x = np.asarray(x_hist, dtype=float)
    y = np.asarray(y_hist, dtype=float)
    if len(x) < 30:
        return {
            "ok": False,
            "status": "No samples",
            "score": 0.0,
            "confidence": 0.0,
            "sample_count": 0,
            "valid_ratio": 0.0,
            "position_std": 0.0,
            "speed_std": 0.0,
            "acc_std": 0.0,
            "reversal_rate": 0.0,
            "jerk_ratio": 0.0,
            "high_freq_ratio": 0.0,
            "peak_speed_ratio": 0.0,
            "reasons": ["insufficient samples"],
        }

    valid_ratio = float(np.mean(np.isfinite(x) & np.isfinite(y)))
    if valid_ratio < MIN_VALID_RATIO:
        return {
            "ok": False,
            "status": "Data lost",
            "score": 0.0,
            "confidence": 0.0,
            "sample_count": int(len(x)),
            "valid_ratio": valid_ratio,
            "position_std": 0.0,
            "speed_std": 0.0,
            "acc_std": 0.0,
            "reversal_rate": 0.0,
            "jerk_ratio": 0.0,
            "high_freq_ratio": 0.0,
            "peak_speed_ratio": 0.0,
            "reasons": ["too many missing points"],
        }

    x = _fill_missing(x)
    y = _fill_missing(y)
    x = _rolling_mean(_rolling_median(x, SMOOTH_MEDIAN_WINDOW), SMOOTH_MEAN_WINDOW)
    y = _rolling_mean(_rolling_median(y, SMOOTH_MEDIAN_WINDOW), SMOOTH_MEAN_WINDOW)

    x = x - np.mean(x)
    y = y - np.mean(y)
    radius = np.hypot(x, y)
    position_std = float(np.std(radius))

    vx = np.diff(x) / FRAME_DT_SEC
    vy = np.diff(y) / FRAME_DT_SEC
    speed = np.hypot(vx, vy)
    if speed.size < 2:
        return {
            "ok": False,
            "status": "Not enough motion points",
            "score": 0.0,
            "confidence": 0.0,
            "sample_count": int(len(x)),
            "valid_ratio": valid_ratio,
            "position_std": position_std,
            "speed_std": 0.0,
            "acc_std": 0.0,
            "reversal_rate": 0.0,
            "jerk_ratio": 0.0,
            "high_freq_ratio": 0.0,
            "peak_speed_ratio": 0.0,
            "reasons": ["insufficient motion points"],
        }

    speed_std = float(np.std(speed))
    reversal_rate = max(_reversal_rate(vx), _reversal_rate(vy))

    ax = np.diff(vx) / FRAME_DT_SEC
    ay = np.diff(vy) / FRAME_DT_SEC
    acc_mag = np.hypot(ax, ay)
    acc_std = float(np.std(acc_mag))
    acc_med = np.median(acc_mag)
    if len(acc_mag) > 0:
        mad = np.median(np.abs(acc_mag - acc_med)) + 1e-9
        jerk_threshold = 4.0 * mad
        jerk_ratio = float(np.mean(np.abs(acc_mag - acc_med) > jerk_threshold))
    else:
        jerk_ratio = 0.0
        jerk_threshold = 0.0

    if speed.size > 3:
        peak_speed_ratio = np.mean(speed > (np.mean(speed) + 2.5 * np.std(speed)))
    else:
        peak_speed_ratio = 0.0

    high_freq_ratio = 0.45 * _band_power_ratio(x) + 0.55 * _band_power_ratio(y)

    pos_score = _lin_norm(position_std, 0.01, 0.04)
    speed_score = _lin_norm(speed_std, 0.02, 0.40)
    acc_score = _lin_norm(acc_std, 0.02, 1.0)
    rev_score = _lin_norm(reversal_rate, 0.03, 0.70)
    hf_score = _lin_norm(high_freq_ratio, 0.06, 0.40)
    jerk_score = _lin_norm(jerk_ratio, 0.06, 0.35)
    peak_score = _lin_norm(peak_speed_ratio, 0.06, 0.45)

    raw_score = (
        0.20 * pos_score
        + 0.16 * speed_score
        + 0.17 * acc_score
        + 0.18 * rev_score
        + 0.13 * hf_score
        + 0.12 * jerk_score
        + 0.14 * peak_score
    ) * 100.0
    confidence = float(100.0 / (1.0 + np.exp(-(raw_score - 50.0) / 8.0)))

    if confidence >= NYSTAGMUS_CONF_HIGH:
        status = "NYSTAGMUS LIKELY"
    elif confidence >= NYSTAGMUS_CONF_LOW:
        status = "ELEVATED / BORDERLINE"
    else:
        status = "STABLE"

    flags = []
    if reversal_rate > 0.20:
        flags.append("oscillatory reversals")
    if acc_std > 0.20:
        flags.append("large acceleration")
    if speed_std > 0.08:
        flags.append("velocity noise")
    if peak_speed_ratio > 0.12:
        flags.append("velocity spikes")
    if high_freq_ratio > 0.20:
        flags.append("high-frequency content")
    if jerk_ratio > 0.12:
        flags.append("jerk bursts")
    if not np.isfinite(jerk_threshold):
        jerk_threshold = 0.0

    return {
        "ok": True,
        "status": status,
        "score": float(raw_score),
        "confidence": confidence,
        "sample_count": int(len(x)),
        "valid_ratio": valid_ratio,
        "position_std": position_std,
        "speed_std": speed_std,
        "acc_std": acc_std,
        "reversal_rate": float(reversal_rate),
        "jerk_ratio": float(jerk_ratio),
        "jerk_threshold": float(jerk_threshold),
        "high_freq_ratio": float(high_freq_ratio),
        "peak_speed_ratio": float(peak_speed_ratio),
        "reasons": flags,
    }


def assess_head_motion(face_x_hist, face_y_hist, eye_scale):
    fx = np.asarray(face_x_hist, dtype=float)
    fy = np.asarray(face_y_hist, dtype=float)
    if len(fx) < 3:
        return {"ok": True, "movement": 0.0}
    dx = np.diff(fx)
    dy = np.diff(fy)
    step = np.sqrt(dx * dx + dy * dy) / max(1.0, eye_scale)
    if len(step) == 0:
        movement = 0.0
    else:
        window = min(HEAD_MOVE_EVAL_WINDOW, len(step))
        movement = float(np.sqrt(np.mean(step[-window:] ** 2)))
    return {
        "ok": movement <= HEAD_MOVE_THRESHOLD_NORM,
        "movement": movement,
    }


def _metric_binary(score):
    return 1 if score >= METRIC_BINARY_THRESHOLD else 0


def _publish_session_result(
    subject_name,
    stop_time,
    arrest_time,
    combined_vals=None,
    left_binary=None,
    right_binary=None,
    left_report=None,
    right_report=None,
    head_motion_warning_count=0,
    max_head_movement=0.0,
    **kwargs,
):
    if combined_vals is None:
        combined_vals = kwargs.get("combind_vals", kwargs.get("combind_val", {}))
    if not HGN_API_ENABLED:
        return
    payload = {
        "session_id": f"hgn-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subject_name": subject_name,
        "stop_time": stop_time,
        "arrest_time": arrest_time,
        "head_warning_count": int(head_motion_warning_count),
        "head_movement_too_much": int(head_motion_warning_count >= 2),
        "max_head_movement": float(max_head_movement),
        "metrics": combined_vals,
        "binary": {"left": left_binary, "right": right_binary},
        "scores": {"left": left_report, "right": right_report},
        "notes": {
            "engine": "hgn_tracker2",
            "binary_threshold": METRIC_BINARY_THRESHOLD,
        },
    }

    if not HGN_PUBLISH_ON_SUCCESS:
        return

    try:
        req = urllib.request.Request(
            f"{HGN_API_URL.rstrip('/')}/api/session/finish",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=HGN_API_TIMEOUT_SEC) as response:
            response_body = response.read().decode("utf-8", errors="ignore")
            print(f"Session saved to local API: {response.status} {response_body}")
    except urllib.error.URLError as exc:
        print(f"Session API unavailable ({exc}); continuing local-only.")
    except Exception as exc:
        print(f"Session publish failed: {exc}")


def summarize_hgn_parameters(x_hist, y_hist, head_motion_index=0.0):
    x = _fill_missing(np.asarray(x_hist, dtype=float))
    y = _fill_missing(np.asarray(y_hist, dtype=float))
    if x.size == 0 or y.size == 0:
        return {"ok": False, "reason": "No samples"}
    valid_ratio = float(np.mean(np.isfinite(x) & np.isfinite(y)))
    if valid_ratio < 0.25:
        return {
            "ok": False,
            "reason": "Too many missing points",
            "valid_ratio": valid_ratio,
            "lack_of_smooth_pursuit": 0.0,
            "nystagmus_prior_to_45": 0.0,
            "distinct_nystagmus_max_deviation": 0.0,
            "vertical_nystagmus": 0.0,
        }

    x = _rolling_mean(_rolling_median(x, SMOOTH_MEDIAN_WINDOW), SMOOTH_MEAN_WINDOW)
    y = _rolling_mean(_rolling_median(y, SMOOTH_MEDIAN_WINDOW), SMOOTH_MEAN_WINDOW)
    x = x - np.mean(x)
    y = y - np.mean(y)

    if len(x) < 3:
        return {"ok": False, "reason": "insufficient samples", "valid_ratio": valid_ratio}

    dx = np.diff(x)
    dy = np.diff(y)
    if len(dx) == 0:
        return {"ok": False, "reason": "insufficient samples", "valid_ratio": valid_ratio}

    vx = dx / FRAME_DT_SEC
    vy = dy / FRAME_DT_SEC
    speed = np.hypot(vx, vy)

    ax = np.diff(vx) / FRAME_DT_SEC
    ay = np.diff(vy) / FRAME_DT_SEC
    if len(ax) > 0:
        acc_mag = np.hypot(ax, ay)
        acc_med = np.median(acc_mag)
        mad = np.median(np.abs(acc_mag - acc_med)) + 1e-9
        jerk_threshold = 4.0 * mad
        jerk_ratio = float(np.mean(np.abs(acc_mag - acc_med) > jerk_threshold))
        acc_std = float(np.std(acc_mag))
    else:
        acc_std = 0.0
        jerk_ratio = 0.0

    reversal_x = _reversal_rate(dx)
    reversal_y = _reversal_rate(dy)
    speed_std = float(np.std(speed))
    edge_x = float(np.max(np.abs(x)))
    edge_y = float(np.max(np.abs(y)))
    edge_score = _lin_norm(max(edge_x, edge_y), 0.12, 0.50)
    high_freq = 0.55 * _band_power_ratio(x) + 0.45 * _band_power_ratio(y)
    head_factor = _lin_norm(float(head_motion_index), 0.0, HEAD_MOVE_THRESHOLD_NORM * 1.25)

    rev_score = _lin_norm((reversal_x + reversal_y) / 2.0, 0.03, 0.40)
    speed_score = _lin_norm(speed_std, 0.02, 0.45)
    acc_score = _lin_norm(acc_std, 0.015, 0.30)
    jerk_score = _lin_norm(jerk_ratio, 0.02, 0.22)
    hf_score = _lin_norm(high_freq, 0.03, 0.40)
    max_step = float(np.max(np.hypot(dx, dy)))
    if (
        speed_std < 0.06
        and acc_std < 0.05
        and hf_score < 0.16
        and edge_score < 0.24
        and max_step < 0.030
    ):
        return {
            "ok": True,
            "valid_ratio": valid_ratio,
            "sample_count": int(len(x)),
            "lack_of_smooth_pursuit": 0.0,
            "nystagmus_prior_to_45": 0.0,
            "distinct_nystagmus_max_deviation": 0.0,
            "vertical_nystagmus": 0.0,
        }
    motion_gate = np.clip(
        0.60 * speed_score
        + 0.20 * acc_score
        + 0.10 * hf_score
        + 0.10 * jerk_score,
        0.0,
        1.0,
    )

    lack_of_smooth_pursuit = float(
        np.clip(
            100.0
            * (
                0.28 * rev_score
                + 0.24 * speed_score
                + 0.22 * acc_score
                + 0.12 * hf_score
                + 0.14 * jerk_score
                + 0.00 * head_factor
            )
            * motion_gate,
            0.0,
            100.0,
        )
    )

    nystagmus_prior_to_45 = float(
        np.clip(
            100.0 * (
                0.74 * lack_of_smooth_pursuit / 100.0
                + 0.16 * hf_score
                + 0.10 * edge_score
            ),
            0.0,
            100.0,
        )
    )

    distinct_nystagmus_max_deviation = float(
        np.clip(
            100.0
            * (
                0.46 * edge_score
                + 0.30 * hf_score
                + 0.18 * jerk_score
                + 0.06 * speed_score
            )
            * motion_gate,
            0.0,
            100.0,
        )
    )

    vertical_speed_std = float(np.std(vy))
    vertical_nystagmus = float(
        np.clip(
            100.0
            * (
                0.55 * _lin_norm(reversal_y, 0.03, 0.40)
                + 0.30 * _lin_norm(vertical_speed_std, 0.02, 0.50)
                + 0.15 * hf_score
            )
            * motion_gate,
            0.0,
            100.0,
        )
    )

    return {
        "ok": True,
        "valid_ratio": valid_ratio,
        "sample_count": int(len(x)),
        "lack_of_smooth_pursuit": lack_of_smooth_pursuit,
        "nystagmus_prior_to_45": nystagmus_prior_to_45,
        "distinct_nystagmus_max_deviation": distinct_nystagmus_max_deviation,
        "vertical_nystagmus": vertical_nystagmus,
    }


def main():
    gc.collect()
    torch.cuda.empty_cache()

    predictor = OwlPredictor(
        "google/owlvit-base-patch32", image_encoder_engine="data/owl_image_encoder_patch32.engine"
    )
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    eye_rois = []
    nose_box = None
    face_center = None
    eye_scale = None

    print("Locking eyes, nose, and face...")
    lock_ok = False
    while not lock_ok:
        ret, frame = cap.read()
        if not ret:
            continue

        small_frame = cv2.resize(frame, FRAME_SIZE)
        image_pil = to_writable_rgb_pil(small_frame)

        eye_rois = detect_eye_rois(predictor, image_pil)
        if len(eye_rois) != 2:
            continue

        face_center = detect_face_centroid(predictor, image_pil)
        if face_center is None:
            # Fallback to midpoint between eyes if face not detected this frame
            eye_l, eye_r = eye_rois
            face_center = (
                (eye_l[0] + eye_l[2] + eye_r[0] + eye_r[2]) / 4.0,
                (eye_l[1] + eye_l[3] + eye_r[1] + eye_r[3]) / 4.0,
            )

        nose_boxes, _ = detect_boxes(predictor, image_pil, "nose")
        if len(nose_boxes) >= 1:
            nose_idx = int(np.argmax([box_area(b) for b in nose_boxes]))
            nose_box = tuple(nose_boxes[nose_idx])
            lock_ok = True
        else:
            # allow lock without nose and still continue using eye-based nose fallback
            lock_ok = True

    if not eye_rois:
        cap.release()
        raise RuntimeError("Could not lock eyes. Try better lighting and closer face distance.")

    if nose_box is None:
        eye_l = eye_rois[0]
        eye_r = eye_rois[1]
        nx = (eye_l[0] + eye_l[2] + eye_r[0] + eye_r[2]) / 4.0
        ny = (eye_l[1] + eye_l[3] + eye_r[1] + eye_r[3]) / 4.0
        nose_box = (int(nx - 8), int(ny + 10), int(nx + 8), int(ny + 24))

    eye_scale = max(20.0, (eye_rois[1][0] + eye_rois[1][2]) / 2 - (eye_rois[0][0] + eye_rois[0][2]) / 2)
    if eye_scale <= 0:
        cap.release()
        raise RuntimeError("Invalid eye geometry for scaling.")

    l_hist, r_hist, l_hist_y, r_hist_y = (
        deque(maxlen=REQUIRED_FRAMES),
        deque(maxlen=REQUIRED_FRAMES),
        deque(maxlen=REQUIRED_FRAMES),
        deque(maxlen=REQUIRED_FRAMES),
    )
    face_x_hist = deque(maxlen=REQUIRED_FRAMES)
    face_y_hist = deque(maxlen=REQUIRED_FRAMES)
    eye_miss_streak = [0, 0]
    last_rel_positions = [None, None]
    nose_miss_streak = 0
    face_miss_streak = 0
    head_motion_warnings = 0
    head_motion_warning_count = 0
    max_head_movement = 0.0

    print("STARTING HGN capture: hold head still for 4s...")

    for f_idx in range(REQUIRED_FRAMES):
        ret, frame = cap.read()
        if not ret:
            continue

        small_frame = cv2.resize(frame, FRAME_SIZE)
        image_pil = to_writable_rgb_pil(small_frame)

        fx = None
        fy = None
        if f_idx % FACE_REFRESH_EVERY == 0:
            detected_face = detect_face_centroid(predictor, image_pil)
            if detected_face is not None:
                fx, fy = detected_face
                face_miss_streak = 0
            else:
                face_miss_streak += 1
                relocked_rois = detect_eye_rois(
                    predictor,
                    image_pil,
                    threshold=max(0.08, DETECTION_THRESHOLD * 0.9),
                )
                if len(relocked_rois) == 2:
                    face_miss_streak = 0
                    ex1, ey1, ex2, ey2 = relocked_rois[0]
                    ex3, ey3, ex4, ey4 = relocked_rois[1]
                    fx = (ex1 + ex2 + ex3 + ex4) / 4.0
                    fy = (ey1 + ey2 + ey3 + ey4) / 4.0
        else:
            if len(face_x_hist) > 0:
                fx = face_x_hist[-1]
                fy = face_y_hist[-1]

        if face_miss_streak > FACE_MISS_RELOCK_LIMIT and face_miss_streak % 30 == 0:
            print(
                f"Face detection miss streak: {face_miss_streak}/{FACE_MISS_RELOCK_LIMIT}; "
                "using fallback tracking."
            )

        if f_idx % NOSE_REFRESH_EVERY == 0:
            nose_boxes, _ = detect_boxes(predictor, image_pil, "nose")
            if len(nose_boxes) > 0:
                nose_idx = int(np.argmax([box_area(b) for b in nose_boxes]))
                nose_box = tuple(nose_boxes[nose_idx])
                nose_miss_streak = 0
            else:
                nose_miss_streak += 1

        if nose_box is None:
            fallback_nose = estimate_nose_from_eyes(eye_rois)
            if fallback_nose is None:
                continue
            nose_box = fallback_nose

        nx1, ny1, nx2, ny2 = nose_box
        if any(v is None for v in (nx1, ny1, nx2, ny2)):
            fallback_nose = estimate_nose_from_eyes(eye_rois)
            if fallback_nose is None:
                continue
            nose_box = fallback_nose
            nx1, ny1, nx2, ny2 = nose_box

        if nose_miss_streak > NOSE_MISS_RELOCK_LIMIT:
            fallback_nose = estimate_nose_from_eyes(eye_rois)
            if fallback_nose is not None:
                nose_box = fallback_nose
                nx1, ny1, nx2, ny2 = nose_box

        if f_idx % NOSE_REFRESH_EVERY == 0 and f_idx > 0 and max(eye_miss_streak) >= RELOCK_EYE_MISSES:
            relocked_rois = detect_eye_rois(predictor, image_pil)
            if relocked_rois:
                eye_rois = relocked_rois
                eye_miss_streak = [0, 0]
                last_rel_positions = [None, None]

        nx = (nx1 + nx2) / 2
        ny = (ny1 + ny2) / 2
        if fx is None:
            fx = nx
            fy = ny

        if fx is not None and fy is not None:
            face_x_hist.append(fx)
            face_y_hist.append(fy)
            head = assess_head_motion(face_x_hist, face_y_hist, eye_scale)
            if not head["ok"]:
                head_motion_warning_count += 1
                head_motion_warnings += 1
                if head_motion_warnings % HEAD_WARNING_PRINT_STRIDE == 0:
                    print(
                        f"HEAD MOTION WARNING x{head_motion_warnings}: {head['movement']:.3f}"
                    )
                max_head_movement = max(max_head_movement, head["movement"])
            else:
                head_motion_warnings = 0

        if len(l_hist) >= REQUIRED_FRAMES and len(r_hist) >= REQUIRED_FRAMES:
            break

        for i, roi in enumerate(eye_rois):
            x1, y1, x2, y2 = roi
            rel_x = float("nan")
            rel_y = float("nan")
            eye_crop = small_frame[y1:y2, x1:x2]
            if eye_crop.size == 0:
                eye_miss_streak[i] += 1
            else:
                pupil = get_pupil_location(eye_crop, is_left_eye=(i == 0))
                if pupil is None:
                    eye_miss_streak[i] += 1
                    last_rel = last_rel_positions[i]
                    if last_rel is not None and eye_miss_streak[i] <= MAX_PUPIL_HOLD_FRAMES:
                        rel_x, rel_y = last_rel
                else:
                    eye_miss_streak[i] = 0
                    px, py = pupil
                    gx, gy = x1 + px, y1 + py
                    rel_x = (gx - nx) / eye_scale
                    rel_y = (gy - ny) / eye_scale
                    last_rel_positions[i] = (rel_x, rel_y)

            if i == 0:
                l_hist.append(rel_x)
                l_hist_y.append(rel_y)
            else:
                r_hist.append(rel_x)
                r_hist_y.append(rel_y)

        if f_idx % 30 == 0:
            print(
                f"Frame {f_idx}/{REQUIRED_FRAMES}: "
                f"L={len(l_hist)} R={len(r_hist)}"
            )

        if f_idx % 60 == 0 and f_idx > 0:
            torch.cuda.empty_cache()

    left_report = summarize_hgn_parameters(
        np.array(l_hist),
        np.array(l_hist_y),
        head_motion_index=max_head_movement,
    )
    right_report = summarize_hgn_parameters(
        np.array(r_hist),
        np.array(r_hist_y),
        head_motion_index=max_head_movement,
    )
    left_valid = left_report.get("ok", False)
    right_valid = right_report.get("ok", False)

    combined_vals = {}
    for key in [
        "lack_of_smooth_pursuit",
        "nystagmus_prior_to_45",
        "distinct_nystagmus_max_deviation",
        "vertical_nystagmus",
    ]:
        valid_vals = []
        if left_valid:
            valid_vals.append(left_report[key])
        if right_valid:
            valid_vals.append(right_report[key])
        combined_vals[key] = float(np.mean(valid_vals)) if valid_vals else 0.0

    left_binary = {
        "lack_of_smooth_pursuit": _metric_binary(left_report["lack_of_smooth_pursuit"]) if left_valid else 0,
        "nystagmus_prior_to_45": _metric_binary(left_report["nystagmus_prior_to_45"]) if left_valid else 0,
        "distinct_nystagmus_max_deviation": _metric_binary(left_report["distinct_nystagmus_max_deviation"]) if left_valid else 0,
    }
    right_binary = {
        "lack_of_smooth_pursuit": _metric_binary(right_report["lack_of_smooth_pursuit"]) if right_valid else 0,
        "nystagmus_prior_to_45": _metric_binary(right_report["nystagmus_prior_to_45"]) if right_valid else 0,
        "distinct_nystagmus_max_deviation": _metric_binary(right_report["distinct_nystagmus_max_deviation"]) if right_valid else 0,
    }

    print("\n" + "=" * 34)
    print(f"Subject Name: {HGN_SUBJECT_NAME}")
    print(f"Stop Time: {HGN_STOP_TIME}")
    print(f"Arrest Time: {HGN_ARREST_TIME}")
    print("HGN FIELD EVALUATION")
    print("-" * 34)
    print(
        "Lack of smooth pursuit: L=%s R=%s"
        % (
            left_binary["lack_of_smooth_pursuit"],
            right_binary["lack_of_smooth_pursuit"],
        )
    )
    print(
        "Nystagmus prior to 45°: L=%s R=%s"
        % (
            left_binary["nystagmus_prior_to_45"],
            right_binary["nystagmus_prior_to_45"],
        )
    )
    print(
        "Distinct nystagmus at max deviation: L=%s R=%s"
        % (
            left_binary["distinct_nystagmus_max_deviation"],
            right_binary["distinct_nystagmus_max_deviation"],
        )
    )
    print(
        "Vertical nystagmus: %.2f/100 (Not validated)"
        % combined_vals["vertical_nystagmus"]
    )
    if head_motion_warning_count >= 2:
        print("Head movement too much, please retest or RESULT VOID.")
    print("=" * 34)

    _publish_session_result(
        subject_name=HGN_SUBJECT_NAME,
        stop_time=HGN_STOP_TIME,
        arrest_time=HGN_ARREST_TIME,
        combined_vals=combined_vals,
        left_binary=left_binary,
        right_binary=right_binary,
        left_report=left_report,
        right_report=right_report,
        head_motion_warning_count=head_motion_warning_count,
        max_head_movement=max_head_movement,
    )

    cap.release()


if __name__ == "__main__":
    main()
