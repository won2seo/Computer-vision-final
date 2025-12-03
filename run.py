# ============================================================
# run.py — Violence Intensity (1~4 단계) 재학습 최적화 버전
# ResNet50 유지 + 학습 속도 개선 + 정확도 향상 버전
# ============================================================

import os
from itertools import chain
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.applications import (
    Xception, ResNet50, InceptionV3, MobileNet,
    VGG19, DenseNet121, InceptionResNetV2, VGG16
)

import BuildModel_basic
import DatasetBuilder

# 시드 고정 (재현성)
tf.random.set_seed(2)
np.random.seed(1)


# ============================================================
# Test Callback
# ============================================================
class TestCallback(Callback):
    def __init__(self, test_data, batch_size):
        super().__init__()
        self.test_data = test_data
        self.test_loss = []
        self.test_acc = []
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, batch_size=self.batch_size, verbose=0)
        self.test_loss.append(loss)
        self.test_acc.append(acc)
        print(f"\n[TEST] epoch={epoch} loss={loss:.4f}, acc={acc:.4f}\n")


# ============================================================
# 🔥 학습 히스토리 그래프 저장
# ============================================================
def save_history_plots(dataset_name, history, out_dir="results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    hist = history.history

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(hist.get("accuracy", []), label="train_acc")
    plt.plot(hist.get("val_accuracy", []), label="val_acc")
    plt.title(f"{dataset_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_accuracy.png"))
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(hist.get("loss", []), label="train_loss")
    plt.plot(hist.get("val_loss", []), label="val_loss")
    plt.title(f"{dataset_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_loss.png"))
    plt.close()

    print(f"[INFO] Plots saved in {out_dir}")


# ============================================================
# Training Function
# ============================================================
def train_eval_network(dataset_name, train_gen, validate_gen,
                       test_x, test_y,
                       seq_len, epochs, batch_size,
                       batch_epoch_ratio, initial_weights, size,
                       cnn_arch, learning_rate,
                       optimizer, cnn_train_type, pre_weights,
                       lstm_conf, len_train, len_valid,
                       dropout, classes,
                       patience_es=15, patience_lr=5,
                       class_weight=None):

    tf.random.set_seed(2)
    np.random.seed(1)

    # 정보 출력
    result = dict(
        dataset=dataset_name,
        cnn_train=cnn_train_type,
        cnn=cnn_arch.__name__,
        lstm=lstm_conf[0].__name__,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        dropout=dropout,
        optimizer=optimizer[0].__name__,
        initial_weights=initial_weights,
        seq_len=seq_len
    )

    print("\n=== Run experiment ===")
    print(result)

    # --------------------------------------------------------
    # 모델 생성
    # --------------------------------------------------------
    model, finetune_cb = BuildModel_basic.build(
        size=size,
        seq_len=seq_len,
        learning_rate=learning_rate,
        optimizer_class=optimizer,
        initial_weights=initial_weights,
        cnn_class=cnn_arch,
        pre_weights=pre_weights,
        lstm_conf=lstm_conf,
        cnn_train_type=cnn_train_type,
        dropout=dropout,
        classes=classes
    )

    steps_per_epoch = max(int(float(len_train) / float(batch_size * batch_epoch_ratio)), 1)
    val_steps = max(int(float(len_valid) / float(batch_size)), 1)

    test_history = TestCallback((test_x, test_y), batch_size=batch_size)

    # --------------------------------------------------------
    # 체크포인트
    # --------------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join("checkpoints", f"{dataset_name}_best_model.h5"),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )

    # --------------------------------------------------------
    # 콜백
    # --------------------------------------------------------
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=patience_es,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-8,
            verbose=1
        ),
        test_history,
        checkpoint
    ]

    if finetune_cb is not None:
        callbacks.append(finetune_cb)

    # --------------------------------------------------------
    # 학습 실행 (class_weight 포함)
    # --------------------------------------------------------
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validate_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        class_weight=class_weight
    )

    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------
    hist = history.history
    result["val_acc"] = max(hist.get("val_accuracy", [0.0]))
    result["val_loss"] = hist["val_loss"][-1]

    result["train_acc"] = max(hist.get("accuracy", [0.0]))
    result["train_loss"] = min(hist["loss"])

    result["test_acc"] = max(test_history.test_acc) if test_history.test_acc else 0.0
    result["test_loss"] = min(test_history.test_loss) if test_history.test_loss else 0.0

    os.makedirs("models", exist_ok=True)
    final_path = os.path.join("models", f"{dataset_name}_final_model.h5")
    model.save(final_path)
    print(f"\n=== ✔ Final model saved at: {final_path} ===")

    save_history_plots(dataset_name, history)

    return result


# ============================================================
# 데이터셋 생성
# ============================================================
def get_generators(dataset_name, dataset_videos, datasets_frames,
                   fix_len, figure_size, force,
                   classes=1, use_aug=False,
                   use_crop=True, crop_dark=None):

    train_path, valid_path, test_path, \
        train_y, valid_y, test_y, \
        avg_length = DatasetBuilder.createDataset(
            dataset_videos, datasets_frames, fix_len, force=force
        )

    if fix_len is not None:
        avg_length = fix_len

    crop_x_y = crop_dark.get(dataset_name, None) if crop_dark else None

    len_train, len_valid = len(train_path), len(valid_path)

    train_gen = DatasetBuilder.data_generator(
        train_path, train_y,
        batch_size=batch_size,
        figure_shape=figure_size,
        seq_length=avg_length,
        use_aug=use_aug,
        use_crop=use_crop,
        crop_x_y=crop_x_y,
        classes=classes
    )

    validate_gen = DatasetBuilder.data_generator(
        valid_path, valid_y,
        batch_size=batch_size,
        figure_shape=figure_size,
        seq_length=avg_length,
        use_aug=use_aug,
        use_crop=use_crop,
        crop_x_y=crop_x_y,
        classes=classes
    )

    test_x, test_y = DatasetBuilder.get_sequences(
        test_path, test_y,
        figure_shape=figure_size,
        seq_length=avg_length,
        classes=classes,
        use_augmentation=False,
        use_crop=use_crop,
        crop_x_y=crop_x_y
    )

    return train_gen, validate_gen, test_x, test_y, avg_length, len_train, len_valid


# ============================================================
# 실험 구동 설정
# ============================================================

datasets_videos = dict(
    violence_intensity=dict(
        violent=os.path.join("data", "violence-detection-dataset", "violent"),
        **{"non-violent": os.path.join("data", "violence-detection-dataset", "non-violent")}
    )
)

crop_dark = dict(
    violence_intensity=None
)

datasets_frames = os.path.join("data", "raw_frames")
res_path = "results"
os.makedirs(res_path, exist_ok=True)

# ============================================================
# 🔥 최적화된 하이퍼파라미터
# ============================================================

figure_size = 224         # ResNet50 권장 입력크기
batch_size = 8            # 속도 개선
fix_len = 32              # 10 → 32 프레임 (1초 단위 행동 반영)
initial_weights = "glorot_uniform"
weights = "imagenet"
force = False

lstm = (tf.keras.layers.LSTM, dict(units=128, return_sequences=False))  # 256→128
classes = 4

cnn_arch = ResNet50       # ✔ ResNet50 유지
learning_rate = 1e-4
optimizer = (RMSprop, {})

cnn_train_type = "static"  # ✔ backbone 완전 고정 (속도↑ 안정성↑)
dropout = 0.5              # 0.6→0.5
use_aug = True

# 🔥 클래스 불균형 해결 (4단계 쏠림 방지)
class_weight = {
    0: 2.0,   # Level 1
    1: 1.7,   # Level 2
    2: 1.3,   # Level 3
    3: 1.0    # Level 4
}


# ============================================================
# 실행 루프
# ============================================================

results = []

for dataset_name, dataset_videos_dict in datasets_videos.items():
    print(f"\n=== Dataset: {dataset_name} ===")

    train_gen, validate_gen, test_x, test_y, seq_len, len_train, len_valid = get_generators(
        dataset_name=dataset_name,
        dataset_videos=dataset_videos_dict,
        datasets_frames=datasets_frames,
        fix_len=fix_len,
        figure_size=figure_size,
        force=force,
        classes=classes,
        use_aug=use_aug,
        use_crop=True,
        crop_dark=crop_dark
    )

    result = train_eval_network(
        epochs=20,
        dataset_name=dataset_name,
        train_gen=train_gen,
        validate_gen=validate_gen,
        test_x=test_x,
        test_y=test_y,
        seq_len=seq_len,
        batch_size=batch_size,
        batch_epoch_ratio=0.5,
        initial_weights=initial_weights,
        size=figure_size,
        cnn_arch=cnn_arch,
        learning_rate=learning_rate,
        optimizer=optimizer,
        cnn_train_type=cnn_train_type,
        pre_weights=weights,
        lstm_conf=lstm,
        len_train=len_train,
        len_valid=len_valid,
        dropout=dropout,
        classes=classes,
        class_weight=class_weight   # ✔ 추가됨
    )

    results.append(result)
    pd.DataFrame(results).to_csv(os.path.join(res_path, "results_datasets.csv"), index=False)

pd.DataFrame(results).to_csv(os.path.join(res_path, "results.csv"), index=False)
print("\n=== Finished. Results saved to results/ ===")
