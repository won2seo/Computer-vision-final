# ============================================================
# BuildModel_basic.py — 기존 구조 유지 + BN freeze 추가 버전
# ============================================================

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization,
    TimeDistributed, GlobalAveragePooling2D, LSTM
)
from tensorflow.keras.optimizers import Adam, RMSprop


# ============================================================
# 🔥 Fine-tuning Callback (기존 유지)
# ============================================================
class FineTuneCallback(tf.keras.callbacks.Callback):
    def __init__(self, base_cnn, unfreeze_epoch=5, unfreeze_layers=30):
        super().__init__()
        self.base_cnn = base_cnn
        self.unfreeze_epoch = unfreeze_epoch
        self.unfreeze_layers = unfreeze_layers
        self.unfrozen = False

    def on_epoch_begin(self, epoch, logs=None):
        if (not self.unfrozen) and (epoch >= self.unfreeze_epoch):
            print(f"\n🔥 Fine-tuning 시작: CNN 마지막 {self.unfreeze_layers}개 레이어 unfreeze\n")
            for layer in self.base_cnn.layers[-self.unfreeze_layers:]:
                layer.trainable = True
            self.unfrozen = True

            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            new_lr = lr * 0.1
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"⚙ Learning rate {lr} -> {new_lr}")


# ============================================================
# 🔥 build() — 수정 포인트 최소화
# ============================================================
def build(size, seq_len, learning_rate,
          optimizer_class,
          initial_weights,
          cnn_class,
          pre_weights,
          lstm_conf,
          cnn_train_type,
          dropout,
          classes):

    input_layer = Input(shape=(seq_len, size, size, 3))

    # CNN 백본 생성 (기존 유지)
    base_cnn = cnn_class(
        include_top=False,
        weights=pre_weights,
        input_shape=(size, size, 3)
    )

    finetune_cb = None

    # ============================================================
    # 🔥 (추가됨) BatchNormalization freeze (BN drift 방지)
    # ============================================================
    for layer in base_cnn.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # ============================================================
    # CNN Train-Type 처리 (기존 유지)
    # ============================================================
    if cnn_train_type == "static":
        base_cnn.trainable = False

    elif cnn_train_type == "retrain":
        base_cnn.trainable = True

    elif cnn_train_type == "static_finetune":
        base_cnn.trainable = False
        finetune_cb = FineTuneCallback(base_cnn, unfreeze_epoch=5, unfreeze_layers=30)

    else:
        print(f"[WARN] Unknown cnn_train_type='{cnn_train_type}', fallback to static")
        base_cnn.trainable = False

    # ============================================================
    # TimeDistributed CNN + GAP (기존 유지)
    # ============================================================
    x = TimeDistributed(base_cnn)(input_layer)
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    # ============================================================
    # LSTM (기존 유지)
    # ============================================================
    lstm_cls, lstm_kwargs = lstm_conf
    if "units" not in lstm_kwargs:
        lstm_kwargs = {**lstm_kwargs, "units": 128}
    x = lstm_cls(**lstm_kwargs)(x)

    # ============================================================
    # Dense Head (기존 유지)
    # ============================================================
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation="relu")(x)

    if classes > 1:
        activation = "softmax"
        loss_func = "categorical_crossentropy"
    else:
        activation = "sigmoid"
        loss_func = "binary_crossentropy"

    predictions = Dense(classes, activation=activation)(x)

    OptimClass, opt_kwargs = optimizer_class
    optimizer = OptimClass(learning_rate=learning_rate, **opt_kwargs)

    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss_func, metrics=["accuracy"])

    print(model.summary())
    return model, finetune_cb
