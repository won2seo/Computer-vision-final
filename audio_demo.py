import streamlit as st
import os
import tempfile
import whisper
import torch
import librosa
import numpy as np
from transformers import BertForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ 설정: 고함 기준값 (0.0 ~ 1.0)
# ==========================================
SHOUTING_THRESHOLD = 0.5 

# ==========================================
# 1. AI 모델 로더 (캐싱으로 속도 최적화)
# ==========================================
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on {device}...")
    
    # 1. Whisper (Medium 모델 권장)
    try:
        stt_model = whisper.load_model("medium", device=device)
    except:
        stt_model = whisper.load_model("small", device=device)
        
    # 2. Unsmile BERT (한국어 욕설 감지)
    model_name = 'smilegate-ai/kor_unsmile'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp_model = BertForSequenceClassification.from_pretrained(model_name)
    
    pipe = TextClassificationPipeline(
        model=nlp_model, 
        tokenizer=tokenizer, 
        device=0 if device == "cuda" else -1, 
        return_all_scores=True
    )
    return stt_model, pipe

# ==========================================
# 2. 분석 핵심 로직
# ==========================================
def analyze_audio(file_path, stt_model, pipe):
    # --- [Step 1] 고함(Shouting) 감지 ---
    # librosa로 소리 크기 분석
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=60) # 최대 60초 분석
        rms = librosa.feature.rms(y=y)[0]
        # 순간 최대 볼륨(상위 1%) 측정
        peak_vol = np.percentile(rms, 99) if len(rms) > 0 else 0
        is_shouting = peak_vol > SHOUTING_THRESHOLD
    except:
        peak_vol = 0
        is_shouting = False

    # --- [Step 2] STT 변환 ---
    result = stt_model.transcribe(
        file_path, language="ko", 
        initial_prompt="욕설, 비속어, 싸움, 거친 표현, 패드립, 고함"
    )
    text = result["text"].strip()
    
    # 환각(Hallucination) 제거
    hallucinations = ["이 대화는 한국어", "MBC 뉴스", "자막 뉴스", "시청해 주셔서"]
    for h in hallucinations:
        text = text.replace(h, "")
    text = text.strip()

    # --- [Step 3] 텍스트 & 상황 판단 ---
    intensity = 0
    actions = []

    # 고함이 감지되면 기본적으로 태그 추가
    if is_shouting:
        actions.append("고함(Shouting)")

    if not text:
        # 대사가 없는데 소리만 지른 경우 (비명 등)
        if is_shouting:
            intensity = 1
            text = "(대사 없음 - 비명/고함 감지)"
        else:
            intensity = 0
            text = "(침묵)"
            actions.append("정상")
            
        return intensity, list(set(actions)), text, peak_vol

    # BERT 분석
    outputs = pipe(text)[0]
    best = max(outputs, key=lambda x: x['score'])
    label = best['label']
    score = best['score']

    # 패드립 키워드 검사
    pad_lip_keywords = ["느금", "니애미", "니애비", "애미", "애비", "창녀", "엠창", "느개비","느금마","니네엄마","니네아빠","너네엄마","너네아빠"]
    is_pad_lip = any(k in text.replace(" ","") for k in pad_lip_keywords)

    # ----------------------------------------
    # 🎚️ 레벨링 로직 (Level 0 ~ 3)
    # ----------------------------------------
    
    # 1. 패드립 / 혐오 표현 (Level 3)
    if is_pad_lip or (label not in ['clean', '욕설', '악플/욕설']):
        intensity = 3
        actions.append("혐오/차별")
        if is_pad_lip: actions.append("패드립")
        
    # 2. 일반 욕설 (Level 1 ~ 2)
    elif label in ['욕설', '악플/욕설']:
        if score < 0.75:
            intensity = 1
            actions.append("비속어")
        else:
            intensity = 2
            actions.append("심한욕설")
            
    # 3. 정상 대화지만 고함을 지른 경우 (Level 1)
    else: # label == 'clean'
        if is_shouting:
            intensity = 1
            actions.append("짜증/분노")
        else:
            intensity = 0
            actions.append("정상대화")

    # 고함 + 욕설이면 강도 상향 조정 (옵션)
    if is_shouting and intensity == 1:
        # 가벼운 욕설 + 고함 -> 심각한 상황으로 볼 수도 있음
        pass 

    return intensity, list(set(actions)), text, peak_vol

# ==========================================
# 3. Streamlit UI 구성
# ==========================================
st.set_page_config(page_title="음성 욕설/고함 탐지기", page_icon="🎤")

st.title("🎤 음성 욕설/고함 탐지기")
st.markdown("오디오를 분석하여 **욕설, 패드립, 고함(Shouting)**을 탐지합니다.")

# 사이드바 상태창
with st.sidebar:
    st.header("시스템 상태")
    with st.spinner("AI 모델 로딩 중..."):
        stt_model, pipe = load_models()
    st.success("✅ AI 모델 준비 완료")
    st.info(f"고함 감지 기준값: {SHOUTING_THRESHOLD}")

# 파일 업로드
uploaded_file = st.file_uploader("파일 업로드 (mp4, mp3, wav)", type=["mp4", "mp3", "wav", "m4a"])

if uploaded_file is not None:
    # 임시 파일 저장
    suffix = ".mp4" if uploaded_file.type.startswith("video") else ".mp3"
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded_file.read())
    file_path = tfile.name

    # 미디어 플레이어
    if suffix == ".mp4":
        st.video(file_path)
    else:
        st.audio(file_path)

    # 분석 버튼
    if st.button("🔍 분석 시작", type="primary"):
        
        with st.spinner('소리 크기와 대화 내용을 분석하고 있습니다...'):
            level, tags, transcript, vol = analyze_audio(file_path, stt_model, pipe)

        st.divider()
        
        # --- 결과 화면 ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🛡️ 최종 판정")
            if level == 0:
                st.success(f"### ✅ Level 0 (안전)")
            elif level == 1:
                st.warning(f"### ⚠️ Level 1 (주의)")
            elif level == 2:
                st.error(f"### 🚨 Level 2 (경고)")
            else:
                st.error(f"### 🚫 Level 3 (심각)")
            
            st.write(f"**감지된 태그:** {', '.join(tags)}")

        with col2:
            st.subheader("📊 세부 지표")
            st.metric("최대 볼륨 (Shouting)", f"{vol:.2f}", help=f"기준값 {SHOUTING_THRESHOLD} 넘으면 고함")
            
            # 볼륨이 높으면 경고 표시
            if vol > SHOUTING_THRESHOLD:
                st.caption("📢 **고함이 감지되었습니다!**")

        st.subheader("📝 대화 내용 (Transcript)")
        st.info(f'"{transcript}"')

    # (선택사항) 임시 파일은 윈도우 특성상 바로 삭제하면 에러날 수 있어 유지하거나 나중에 삭제