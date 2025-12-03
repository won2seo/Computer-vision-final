import os
import cv2
import glob
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

corner_keys = ["Center", "Left_up", "Left_down", "Right_up", "Right_down"]

Debug_Print_AUG = False


# ============================================================
# 🔥 Augmentation 함수
# ============================================================
def augment_frame(arr):
    """
    arr: float32, [H, W, 3], 값 범위 [0, 1]
    간단한 augmentation(좌우 반전, 밝기 변화, 노이즈)을 적용
    """
    # 좌우 반전
    if random.random() < 0.5:
        arr = np.fliplr(arr)

    # 밝기 조절 (0.8 ~ 1.2 배)
    if random.random() < 0.5:
        factor = 0.8 + random.random() * 0.4
        arr = np.clip(arr * factor, 0.0, 1.0)

    # 가우시안 노이즈 약간 추가
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.02, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)

    return arr


def extract_and_save_frames(video_path, video_id, figures_root,
                            fix_len=None, skip_frames=None):
    """
    비디오에서 프레임을 추출해서
    data/raw_frames/{video_id}/frame_0.jpg 이런 식으로 저장.
    """
    video_figures_path = os.path.join(figures_root, video_id)
    os.makedirs(video_figures_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fix_len is not None and fix_len > 0 and total_frames > 0:
        # 전체 프레임 중 fix_len 개를 균등하게 뽑기 위한 step
        step = max(total_frames // fix_len, 1)
    else:
        step = 1
        fix_len = None  # 제한 두지 않음

    seq_len = 0
    idx = 0
    frame_files = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            frame_file = os.path.join(video_figures_path, f"frame_{seq_len}.jpg")
            cv2.imwrite(frame_file, frame)
            frame_files.append(frame_file)
            seq_len += 1
            if fix_len is not None and seq_len >= fix_len:
                break

        idx += 1

    cap.release()

    video_images = dict(
        images_path=video_figures_path,
        name=video_id,
        images_files=frame_files,
        sequence_length=seq_len
    )
    return video_images


def createDataset(datasets_video_path, figure_output_path, fix_len, force=False):
    """
    기존 인터페이스 유지:
    - datasets_video_path: dict
        {
          "violent": "data/violence-detection-dataset/violent",
          "non-violent": "data/violence-detection-dataset/non-violent"
        }
    - figure_output_path: "data/raw_frames"
    - fix_len: 시퀀스 길이 (예: 20)

    반환:
    train_path, valid_path, test_path,
    train_y, valid_y, test_y,
    avg_length
    """
    labels_csv = os.path.join("data", "violence_intensity_labels.csv")
    labels_df = pd.read_csv(labels_csv)

    # video_id -> intensity (0~3)
    video2intensity = dict(zip(labels_df["video_id"], labels_df["intensity"]))

    videos_seq_length = []
    videos_frames_paths = []
    videos_labels = []

    for cls_key, cls_root in datasets_video_path.items():
        # cls_key: "violent" 또는 "non-violent"
        for cam in ["cam1", "cam2"]:
            cam_dir = os.path.join(cls_root, cam)
            if not os.path.isdir(cam_dir):
                continue

            for filename in sorted(os.listdir(cam_dir)):
                if not filename.lower().endswith(".mp4"):
                    continue

                base = os.path.splitext(filename)[0]  # "1", "2", ...
                if cls_key == "violent":
                    prefix = "violent"
                else:
                    # CSV에서 nonviolent_ 로 만들었으므로 하이픈 없는 이름 사용
                    prefix = "nonviolent"

                video_id = f"{prefix}_{cam}_{base}"

                if video_id not in video2intensity:
                    # 라벨 CSV에 없는 경우 스킵
                    continue

                label = int(video2intensity[video_id])

                # 프레임 저장 경로
                video_figures_path = os.path.join(figure_output_path, video_id)
                summary_pkl = os.path.join(video_figures_path, "video_summary.pkl")

                if os.path.isfile(summary_pkl) and not force:
                    with open(summary_pkl, "rb") as f:
                        video_images = pickle.load(f)
                else:
                    video_path = os.path.join(cam_dir, filename)
                    video_images = extract_and_save_frames(
                        video_path, video_id, figure_output_path, fix_len=fix_len
                    )
                    video_images["label"] = label
                    os.makedirs(video_figures_path, exist_ok=True)
                    with open(summary_pkl, "wb") as f:
                        pickle.dump(video_images, f, pickle.HIGHEST_PROTOCOL)

                videos_seq_length.append(video_images["sequence_length"])
                videos_frames_paths.append(video_images["images_path"])
                videos_labels.append(video_images["label"])

    avg_length = int(float(sum(videos_seq_length)) / max(len(videos_seq_length), 1))

    # stratify로 강도 분포 유지
    train_path, test_path, train_y, test_y = train_test_split(
        videos_frames_paths, videos_labels, test_size=0.20,
        random_state=42, stratify=videos_labels
    )
    train_path, valid_path, train_y, valid_y = train_test_split(
        train_path, train_y, test_size=0.20,
        random_state=42, stratify=train_y
    )

    return train_path, valid_path, test_path, train_y, valid_y, test_y, avg_length


def frame_loader(frames, figure_shape, to_norm=True):
    X = []
    for f in frames:
        img = load_img(f, target_size=(figure_shape, figure_shape))
        arr = img_to_array(img)
        if to_norm:
            arr = arr.astype("float32") / 255.0
        X.append(arr)
    return np.array(X, dtype="float32")


def crop_img(img, figure_shape, percentage=0.8, corner="Center"):
    """
    간단한 크롭 함수. (corner는 Center만 실질적으로 사용)
    """
    h, w, c = img.shape
    target = figure_shape

    # 중앙 크롭
    new_h = int(h * percentage)
    new_w = int(w * percentage)
    y_start = max((h - new_h) // 2, 0)
    x_start = max((w - new_w) // 2, 0)
    cropped = img[y_start:y_start + new_h, x_start:x_start + new_w, :]
    resized = cv2.resize(cropped, (target, target))
    return resized.astype(np.float32)


def natural_sort(l):
    import re
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [
        convert(c) for c in re.split("([0-9]+)", key)
    ]
    return sorted(l, key=alphanum_key)


def get_sequences(data_paths, labels, figure_shape, seq_length,
                  classes=1, use_augmentation=False, use_crop=True, crop_x_y=None):
    """
    test set용: 모든 시퀀스를 한 번에 메모리로 로드.
    test에서는 augmentation 사용 X (use_augmentation 미사용).
    """
    X, y = [], []
    for path, label in zip(data_paths, labels):
        frames = natural_sort(
            glob.glob(os.path.join(path, "frame_*.jpg"))
        )
        if len(frames) == 0:
            continue

        # 뒤에서 seq_length개 사용 (부족하면 앞에서부터 채움)
        if len(frames) >= seq_length:
            frames = frames[-seq_length:]
        else:
            # 부족한 만큼 첫 프레임을 반복해서 padding
            frames = [frames[0]] * (seq_length - len(frames)) + frames

        seq = frame_loader(frames, figure_shape, to_norm=True)
        X.append(seq)
        y.append(label)

    X = np.array(X, dtype="float32")
    if classes > 1:
        y = to_categorical(np.array(y), num_classes=classes)
    else:
        y = np.array(y)
    return X, y


def data_generator(data_paths, labels, batch_size, figure_shape, seq_length,
                   use_aug, use_crop, crop_x_y, classes=1):
    """
    train/validation용 generator
    use_aug=True일 때 위에서 정의한 augment_frame을 사용해
    데이터 증강을 수행.
    """
    n = len(data_paths)
    while True:
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        for start in range(0, n, batch_size):
            batch_idx = idxs[start:start + batch_size]
            X_batch, y_batch = [], []
            for i in batch_idx:
                path = data_paths[i]
                label = labels[i]

                frames = natural_sort(
                    glob.glob(os.path.join(path, "frame_*.jpg"))
                )
                if len(frames) == 0:
                    continue

                if len(frames) >= seq_length:
                    frames_sel = frames[-seq_length:]
                else:
                    frames_sel = [frames[0]] * (seq_length - len(frames)) + frames

                imgs = []
                for f in frames_sel:
                    img = load_img(f, target_size=(figure_shape, figure_shape))
                    arr = img_to_array(img)
                    arr = arr.astype("float32") / 255.0

                    # 🔥 train/val에서만 augmentation 적용
                    if use_aug:
                        arr = augment_frame(arr)

                    imgs.append(arr)
                X_batch.append(np.array(imgs, dtype="float32"))
                y_batch.append(label)

            if len(X_batch) == 0:
                continue

            X_batch = np.array(X_batch, dtype="float32")
            if classes > 1:
                y_batch = to_categorical(np.array(y_batch), num_classes=classes)
            else:
                y_batch = np.array(y_batch)

            yield X_batch, y_batch


def data_generator_files(data, labels, batch_size):
    """
    (안 쓰고 있지만, 혹시 기존 코드에서 사용할 수도 있으니 유지)
    """
    while True:
        indexes = np.arange(len(data))
        np.random.shuffle(indexes)
        select_indexes = indexes[:batch_size]
        X = [data[i] for i in select_indexes]
        y = [labels[i] for i in select_indexes]
        yield X, y

