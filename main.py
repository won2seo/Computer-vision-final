import os
import glob
import pandas as pd
import whisper
import torch
import librosa
import numpy as np
from transformers import BertForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import warnings

# 불필요한 경고 메시지 숨기기
warnings.filterwarnings("ignore")

class AudioDatasetGenerator:
    def __init__(self):
        print("🔄 [Init] 모델을 로드하는 중입니다... (Text + Audio Volume 분석)")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   👉 사용 장치: {self.device}")

        # 1. Whisper 모델 (Medium)
        try:
            self.stt_model = whisper.load_model("medium", device=self.device)
            print("   ✅ Whisper 'Medium' 모델 로드 성공")
        except Exception as e:
            print(f"   ⚠️ 메모리 부족으로 'Small' 모델로 대체합니다. ({e})")
            self.stt_model = whisper.load_model("small", device=self.device)

        # 2. Unsmile BERT (욕설 텍스트 감지)
        model_name = 'smilegate-ai/kor_unsmile'
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.nlp_model = BertForSequenceClassification.from_pretrained(model_name)
            
            self.pipe = TextClassificationPipeline(
                model=self.nlp_model, 
                tokenizer=self.tokenizer, 
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            print("   ✅ Unsmile BERT 모델 로드 성공")
        except Exception as e:
            print(f"❌ BERT 모델 로드 실패: {e}")
            exit()

    def detect_shouting(self, file_path, threshold=0.5):
        """
        🔊 오디오의 볼륨(RMS 에너지)을 분석해 고함/비명 여부 판단
        threshold: 고함으로 판단할 기준 볼륨 (0.0 ~ 1.0)
        """
        try:
            # 빠른 분석을 위해 최대 60초까지만 로드
            y, sr = librosa.load(file_path, sr=16000, duration=60)
            
            # 소리 크기(RMS) 계산
            rms = librosa.feature.rms(y=y)[0]
            
            if len(rms) == 0:
                return False, 0.0

            # 전체 평균이 아니라, '순간 최대 볼륨'을 체크
            max_vol = np.max(rms)
            
            # 기준값보다 크면 고함으로 판단
            is_shouting = max_vol > threshold
            return is_shouting, max_vol
            
        except Exception as e:
            print(f"   ⚠️ 볼륨 분석 실패: {e}")
            return False, 0.0

    def analyze_file(self, file_path):
        filename = os.path.basename(file_path)
        print(f"\n📂 분석 중인 파일: {filename}")
        
        # [Step 1] 고함(Shouting) 감지 먼저 실행
        is_shouting, vol_score = self.detect_shouting(file_path)
        if is_shouting:
            print(f"   🔊 [고함 감지] 최대 볼륨: {vol_score:.2f} (기준치 초과)")
        
        try:
            # [Step 2] STT 변환
            result = self.stt_model.transcribe(
                file_path, 
                language="ko", 
                initial_prompt="욕설, 비속어, 싸움, 거친 표현, 패드립, 고함"
            )
            text = result["text"].strip()

            # 환각 제거
            hallucinations = [
                "이 대화는 한국어", "이 대화는 욕설", "MBC 뉴스", "자막 뉴스", "시청해 주셔서"
            ]
            for h in hallucinations:
                if h in text:
                    text = text.replace(h, "").strip()

            print(f"🗣️ 변환된 텍스트: \"{text}\"")
            
        except Exception as e:
            print(f"❌ 오디오 변환 오류: {e}")
            return None

        if not text:
            # 텍스트는 없는데 소리만 지른 경우 (비명)
            if is_shouting:
                print("⚠️ 대사 없음 + 고함 감지 -> Level 1 부여")
                return {
                    "audio_id": os.path.splitext(filename)[0],
                    "actions": '"shouting,scream"',
                    "intensity": 1
                }
            else:
                print("⚠️ 유효한 음성이 감지되지 않았습니다.")
                return {
                    "audio_id": os.path.splitext(filename)[0], 
                    "actions": '"silence"', 
                    "intensity": 0
                }

        # [Step 3] 텍스트 감정/욕설 분석
        outputs = self.pipe(text)[0]
        best = max(outputs, key=lambda x: x['score'])
        label = best['label']
        score = best['score']

        # [Step 4] 데이터셋 라벨링 로직
        intensity = 0
        actions = []

        # 패드립 감지
        pad_lip_keywords = ["느금", "니애미", "니애비", "애미", "애비", "느개비", "창녀", "느검", "엠창","느금마","니네엄마","니네아빠","너네엄마","너네아빠"]
        is_pad_lip = any(keyword in text.replace(" ", "") for keyword in pad_lip_keywords)

        # 1. 패드립/혐오 표현 (최우선 Level 3)
        if is_pad_lip or (label not in ['clean', '욕설', '악플/욕설']):
            intensity = 3
            actions.append("hate_speech")
            if is_pad_lip: actions.append("parental_insult")
            
            label_map = {
                '여성/가족': 'gender_bias', '남성': 'gender_bias',
                '성소수자': 'sexual_minority', '인종/국적': 'racism',
                '연령': 'ageism', '지역': 'regional_bias', '종교': 'religious_bias'
            }
            if label in label_map: actions.append(label_map[label])

        # 2. 일반 욕설
        elif label in ['욕설', '악플/욕설']:
            if score < 0.75:
                intensity = 1
                actions.append("slang")
            else:
                intensity = 2
                actions.append("curse")
                actions.append("insult")
        
        # 3. 정상 대화
        else: # label == 'clean'
            intensity = 0
            actions.append("talk")

        # 🔥 [Final] 고함(Shouting) 반영 로직
        if is_shouting:
            actions.append("shouting")
            # 욕설이 없어도 소리를 질렀으면 최소 Level 1 (짜증/화남)
            if intensity == 0:
                intensity = 1
                actions.append("annoyance")
            # 이미 욕설(Level 1)인데 소리까지 질렀으면 Level 2로 격상 가능 (선택사항)
            # elif intensity == 1:
            #     intensity = 2 

        actions = list(set(actions))
        actions_str = ",".join(actions)

        print(f"📊 상세 분석:")
        print(f" - 감지된 라벨: {label} (확률: {score*100:.1f}%)")
        print(f" - 고함 여부: {is_shouting}")
        print(f" - 최종 강도: Level {intensity}")
        print(f" - 생성된 태그: {actions_str}")

        return {
            "audio_id": os.path.splitext(filename)[0],
            "actions": f'"{actions_str}"',
            "intensity": intensity
        }

    def process_folder(self, input_folder, output_csv):
        extensions = ['*.mp4', '*.mp3', '*.wav', '*.m4a']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(input_folder, ext)))
        
        files = sorted(files)
        total = len(files)
        
        print(f"\n🚀 총 {total}개의 파일을 처리합니다.")
        print(f"💾 결과 저장 경로: {output_csv}")

        dataset = []

        for idx, file_path in enumerate(files):
            print(f"\n{'='*60}")
            print(f"Progress: [{idx+1}/{total}]")
            
            row = self.analyze_file(file_path)
            
            if row:
                dataset.append(row)

        if dataset:
            df = pd.DataFrame(dataset)
            df = df[["audio_id", "actions", "intensity"]]
            try:
                df.to_csv(output_csv, index=False, encoding='utf-8-sig')
                print(f"\n{'='*60}")
                print(f"🎉 모든 작업 완료! '{output_csv}' 파일이 생성되었습니다.")
                print(df.head())
            except PermissionError:
                print(f"\n❌ [오류] '{output_csv}' 파일이 열려있습니다. 닫고 다시 실행해주세요.")
        else:
            print("\n⚠️ 저장할 데이터가 없습니다.")

# =========================================================
# ▶️ 실행 설정
# =========================================================
if __name__ == "__main__":
    # 1. 동영상이 들어있는 폴더 경로
    INPUT_FOLDER = r"C:\Users\user\.spyder-py3"
    
    # 2. 결과를 저장할 파일 이름
    OUTPUT_FILE = "audio_profanity_dataset.csv"

    if os.path.exists(INPUT_FOLDER):
        generator = AudioDatasetGenerator()
        generator.process_folder(INPUT_FOLDER, OUTPUT_FILE)
    else:
        print(f"❌ 폴더를 찾을 수 없습니다: {INPUT_FOLDER}")