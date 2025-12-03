import os
import glob
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence

# =========================================================
# ⚙️ 설정 (Settings)
# =========================================================
CSV_PATH = "audio_profanity_dataset.csv"       # 데이터셋 CSV 파일 경로
AUDIO_FOLDER = r"C:\Users\user\.spyder-py3" # 오디오/비디오 파일이 있는 폴더
MODEL_SAVE_PATH = "audio_intensity_model.h5"

# 오디오 처리 설정
SAMPLE_RATE = 16000
DURATION = 5  # 5초 (모든 오디오를 5초로 맞춤)
N_MELS = 64   # 스펙트로그램 높이
MAX_LEN = int(SAMPLE_RATE * DURATION / 512) # 시간축 길이 계산 (약 157)

CLASSES = 4   # 강도 (0, 1, 2, 3)

# =========================================================
# 1. 데이터 전처리 함수 (Audio -> Image)
# =========================================================
def preprocess_audio(file_path):
    """오디오 파일을 읽어서 멜-스펙트로그램(이미지 형태)으로 변환"""
    try:
        # 1. 오디오 로드 (5초 길이로 고정)
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # 2. 길이가 짧으면 채우고(Padding), 길면 자름(Truncate)
        if len(y) < SAMPLE_RATE * DURATION:
            padding = SAMPLE_RATE * DURATION - len(y)
            y = np.pad(y, (0, padding), mode='constant')
        else:
            y = y[:SAMPLE_RATE * DURATION]
            
        # 3. 멜 스펙트로그램 변환 (소리의 지문 만들기)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 4. 정규화 (0~1 사이 값으로 변환)
        min_val = mel_spec_db.min()
        max_val = mel_spec_db.max()
        norm_spec = (mel_spec_db - min_val) / (max_val - min_val + 1e-6)
        
        # 5. 차원 추가 (CNN 입력 형태: 높이 x 너비 x 1)
        # 결과 shape: (64, 157, 1)
        return norm_spec[..., np.newaxis]
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# =========================================================
# 2. 데이터 제너레이터 (메모리 효율적으로 학습)
# =========================================================
class DataGenerator(Sequence):
    def __init__(self, df, audio_folder, batch_size=16, shuffle=True):
        self.df = df
        self.audio_folder = audio_folder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        # 배치를 위한 인덱스 선택
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        X = []
        y = []
        
        for k in indexes:
            row = self.df.iloc[k]
            file_id = str(row['audio_id'])
            label = int(row['intensity'])
            
            # 파일 찾기 (확장자가 mp4, mp3, wav 중 무엇인지 모르므로 검색)
            search_path = os.path.join(self.audio_folder, f"{file_id}.*")
            found_files = glob.glob(search_path)
            
            if not found_files:
                # 파일을 못 찾으면 건너뜀 (0값으로 대체하거나 스킵)
                continue
                
            file_path = found_files[0] # 첫 번째 매칭 파일 사용
            
            # 전처리
            data = preprocess_audio(file_path)
            if data is not None and data.shape == (N_MELS, MAX_LEN+1, 1): # shape 보정
                 X.append(data)
                 y.append(label)
            elif data is not None:
                # 크기가 안 맞으면 리사이즈
                import cv2
                resized = cv2.resize(data, (MAX_LEN, N_MELS))
                X.append(resized[..., np.newaxis])
                y.append(label)

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# =========================================================
# 3. 모델 설계 (CNN)
# =========================================================
def build_model(input_shape, num_classes):
    model = Sequential([
        # 첫 번째 합성곱 층
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        
        # 두 번째 합성곱 층
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # 세 번째 합성곱 층
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        
        # 출력층 (0, 1, 2, 3 분류)
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', # 0,1,2,3 정수 라벨용
                  metrics=['accuracy'])
    return model

# =========================================================
# 4. 메인 실행부
# =========================================================
if __name__ == "__main__":
    # 1. CSV 로드
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV 파일이 없습니다: {CSV_PATH}")
        exit()
        
    df = pd.read_csv(CSV_PATH)
    print(f"📊 전체 데이터 개수: {len(df)}개")
    
    # 데이터가 너무 적으면 학습 불가
    if len(df) < 10:
        print("⚠️ 데이터가 너무 적습니다. 최소 10개 이상의 파일을 준비해주세요.")
        exit()

    # 2. 학습/검증 데이터 분리 (8:2)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=None)
    
    print(f"   - 학습용: {len(train_df)}개")
    print(f"   - 검증용: {len(val_df)}개")

    # 3. 제너레이터 생성
    train_gen = DataGenerator(train_df, AUDIO_FOLDER, batch_size=8)
    val_gen = DataGenerator(val_df, AUDIO_FOLDER, batch_size=8)
    
    # 4. 모델 생성
    # 입력 shape 계산 (자동) -> (64, 157, 1) 정도 예상
    # 1. mp4 파일만 찾도록 변경
    sample_files = glob.glob(os.path.join(AUDIO_FOLDER, "*.mp4"))

    # (만약 mp3를 쓴다면 "*.mp3"로 바꾸거나, 둘 다 찾게 해야 함)
    if not sample_files:
        sample_files = glob.glob(os.path.join(AUDIO_FOLDER, "*.mp3"))

    if not sample_files:
        print("❌ 폴더에 오디오/비디오 파일이 하나도 없습니다!")
        exit()

    # 2. 찾은 파일 중 첫 번째 것으로 형태 파악
    dummy_data = preprocess_audio(sample_files[0])
    input_shape = dummy_data.shape
    print(f"🧠 모델 입력 형태: {input_shape}")
    
    model = build_model(input_shape, CLASSES)
    model.summary()

    # 5. 학습 시작
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("\n🚀 학습을 시작합니다... (Ctrl+C로 중단 가능)")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30, # 30번 반복 학습
        callbacks=[checkpoint, early_stop]
    )

    print(f"\n✅ 학습 완료! 모델이 저장되었습니다: {MODEL_SAVE_PATH}")
    
    # 1. 학습이 끝난 후 무조건 강제로 저장하기
    model.save(MODEL_SAVE_PATH)
    
    # 2. 파일이 어디에 저장됐는지 정확한 주소를 알려주기
    import os
    print("\n" + "="*50)
    print(f"🎉 파일 생성 완료!")
    print(f"📂 파일 위치: {os.path.abspath(MODEL_SAVE_PATH)}")
    print("="*50 + "\n")