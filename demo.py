import os
import cv2
import numpy as np
import subprocess
import time
import torch
import librosa
import whisper
from transformers import BertForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from tensorflow.keras.models import load_model


# ============================================================
# Paths & Constants
# ============================================================
VIDEO_PATH = "input.mp4"
VIDEO_MODEL_PATH = "checkpoints/violence_intensity_best_model.h5"
OUTPUT_PATH = "output_demo.mp4"
FINAL_OUTPUT = "output(output).mp4"

SEQ_LEN = 32
IMG_SIZE = 224
GRAPH_AREA = 180


# ============================================================
# Load Whisper + BERT
# ============================================================
print("[INFO] Loading audio analysis models (Whisper + BERT)...")

device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    stt_model = whisper.load_model("medium", device=device)
except:
    stt_model = whisper.load_model("small", device=device)

tokenizer = AutoTokenizer.from_pretrained("smilegate-ai/kor_unsmile")
bert_model = BertForSequenceClassification.from_pretrained("smilegate-ai/kor_unsmile")
bert_pipe = TextClassificationPipeline(
    model=bert_model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
    return_all_scores=True
)

print("[INFO] Whisper & BERT Ready.\n")


# ============================================================
# Audio Analysis Function
# ============================================================
def analyze_audio(file_path):
    SHOUT_THRESHOLD = 0.5

    try:
        y, sr = librosa.load(file_path, sr=16000)
        rms = librosa.feature.rms(y=y)[0]
        peak = np.percentile(rms, 99) if len(rms) else 0
        shouting = peak > SHOUT_THRESHOLD
    except:
        peak = 0
        shouting = False

    result = stt_model.transcribe(file_path, language="ko")
    text = result["text"].strip()

    for r in ["이 대화는 한국어", "뉴스", "자막"]:
        text = text.replace(r, "")
    text = text.strip()

    if not text:
        return (1 if shouting else 0), ["shouting" if shouting else "silence"], "(no text)"

    outputs = bert_pipe(text)[0]
    best = max(outputs, key=lambda x: x["score"])
    label, score = best["label"], best["score"]

    intensity = 0
    tags = []

    pad_terms = ["느금", "니애미", "니애비", "느개비", "창녀", "엠창"]
    is_pad = any(p in text.replace(" ", "") for p in pad_terms)

    if is_pad:
        intensity = 3
        tags.append("hate/abuse")

    elif label in ["욕설", "악플/욕설"]:
        intensity = 2 if score >= 0.75 else 1
        tags.append("profanity")

    if shouting and intensity == 0:
        intensity = 1
        tags.append("shouting")

    if not tags:
        tags = ["normal"]

    return intensity, tags, text


# ============================================================
# Load Video Model
# ============================================================
print("[INFO] Loading video model...")
video_model = load_model(VIDEO_MODEL_PATH)
print("[INFO] Video model ready.\n")


def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame.astype(np.float32) / 255.0


def predict_clip(clip_frames):
    x = np.expand_dims(np.array(clip_frames), axis=0)
    preds = video_model.predict(x, verbose=0)[0]
    level = np.argmax(preds) + 1
    return level, preds


# ============================================================
# Open Video + Progress Variables
# ============================================================
print("[INFO] Opening video...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("[ERROR] Cannot open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
TOTAL_H = h + GRAPH_AREA

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, TOTAL_H))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
processed_frames = 0
start_time = time.time()


# ============================================================
# Audio analysis
# ============================================================
print("[INFO] Analyzing audio...")
audio_level, audio_tags, audio_text = analyze_audio(VIDEO_PATH)
print(f"[AUDIO] Level={audio_level}, Tags={audio_tags}")


# ============================================================
# Colors
# ============================================================
risk_colors = {
    1: (0, 255, 0),
    2: (0, 255, 255),
    3: (0, 165, 255),
    4: (0, 0, 255)
}


# ============================================================
# MAIN LOOP
# ============================================================
frame_buf = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------------------
    # Progress bar
    # -----------------------------------------
    processed_frames += 1
    elapsed = time.time() - start_time
    fps_proc = processed_frames / elapsed if elapsed > 0 else 0
    remain = max(total_frames - processed_frames, 0)
    eta = remain / fps_proc if fps_proc > 0 else 0

    print(
        f"\r[PROGRESS] {processed_frames}/{total_frames} "
        f"({processed_frames/total_frames*100:.2f}%) | "
        f"FPS: {fps_proc:.2f} | ETA: {eta:.1f}s",
        end=""
    )

    # -----------------------------------------
    # Maintain buffer
    # -----------------------------------------
    frame_buf.append(preprocess_frame(frame))
    if len(frame_buf) > SEQ_LEN:
        frame_buf.pop(0)

    # Not enough frames yet → show raw frame
    if len(frame_buf) < SEQ_LEN:
        canvas = np.zeros((TOTAL_H, w, 3), np.uint8)
        canvas[:h] = frame
        out.write(canvas)
        continue

    # -----------------------------------------
    # Predict video violence level
    # -----------------------------------------
    video_level, softmax = predict_clip(frame_buf)

    # ----------------------------------------------------------
    # ★ Level 4 남발 방지
    # ----------------------------------------------------------
    if video_level == 4 and softmax[video_level - 1] < 0.90:
        video_level = 3

    # -----------------------------------------
    # Final Level (video 70% + audio 30%)
    # -----------------------------------------
    if video_level == 4:
        final_level = 4
    elif video_level == 3:
        final_score = video_level * 0.9 + audio_level * 0.1
        final_level = round(final_score)
    else:
        final_score = video_level * 0.7 + audio_level * 0.3
        final_level = round(final_score)

    final_level = min(max(final_level, 1), 4)

    reserved = frame.copy()
    cv2.rectangle(reserved, (0, 0), (w - 1, h - 1), risk_colors[final_level], 8)

    graph = np.zeros((GRAPH_AREA, w, 3), dtype=np.uint8)

    cv2.putText(graph, f"Video Level: {video_level}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    cv2.putText(graph, f"Audio Level: {audio_level} ({','.join(audio_tags)})",
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,255,200), 2)

    cv2.putText(graph, f"FINAL LEVEL : {final_level}",
                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.3, risk_colors[final_level], 3)

    canvas = np.zeros((TOTAL_H, w, 3), dtype=np.uint8)
    canvas[:h] = reserved
    canvas[h:] = graph

    out.write(canvas)


cap.release()
out.release()
print("\n[INFO] Video processing complete.")

# ============================================================
# Merge audio with FFmpeg
# ============================================================
print("[INFO] Merging audio...")
cmd = f'ffmpeg -i "{OUTPUT_PATH}" -i "{VIDEO_PATH}" -c copy -map 0:v -map 1:a -y "{FINAL_OUTPUT}"'
subprocess.run(cmd, shell=True)

print(f"[INFO] Completed: {FINAL_OUTPUT}")
