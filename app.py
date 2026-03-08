import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (윈도우용 맑은 고딕)
plt.rc('font', family='Malgun Gothic')

# ---------------------------------------------------------
# 1. 귀농 vs 귀촌 분리 데이터 준비
# ---------------------------------------------------------
data = {
    '연도': [2020, 2021, 2022, 2023, 2024],
    '귀농인구(명)': [12489, 14461, 12660, 10307, 10710],
    '귀촌인구(명)': [477122, 515434, 421106, 400600, 422789]
}
df = pd.DataFrame(data)

# ---------------------------------------------------------
# 2. 인공지능 예측 모델 (귀농/귀촌 개별 학습)
# ---------------------------------------------------------
X = df[['연도']]

# 선형 회귀 모델 학습
model_farming = LinearRegression()
model_farming.fit(X, df['귀농인구(명)'])

model_rural = LinearRegression()
model_rural.fit(X, df['귀촌인구(명)'])

# 2026년 예측
future_year = pd.DataFrame({'연도': [2026]})
pred_farming_2026 = int(model_farming.predict(future_year)[0])
pred_rural_2026 = int(model_rural.predict(future_year)[0])

# ---------------------------------------------------------
# 3. Streamlit 웹 앱 화면(UI) 구성
# ---------------------------------------------------------
st.title("🌾 2026 귀농 vs 귀촌 분리 예측 및 심층 분석 앱")
st.write("귀농(농업 목적)과 귀촌(거주/기타 목적)의 트렌드를 분리하여 분석하고 2026년을 예측합니다.")

# 1. 과거 통계 및 의미
st.header("📊 최근 5년 통계 수치 및 의미")
st.dataframe(df)

# 2. 예측 수치 및 결과
st.header("🔮 2026년 인구 예측 수치")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="2026년 예상 귀농인구", value=f"{pred_farming_2026:,}명", delta="장기적 하락 추세 반영")
with col2:
    st.metric(label="2026년 예상 귀촌인구", value=f"{pred_rural_2026:,}명", delta="장기적 하락 추세 반영")

st.divider()

# 3. 핵심: 예측의 주요 근거 (사용자 요청 사항 반영)
st.header("🧠 2026년 추가 감소 예측의 확실한 근거")
st.warning("""
**Q. 2024년에 수치가 반등했는데, 왜 2026년 예측은 '감소'로 나왔을까요?**

이 앱에 적용된 '선형 회귀(Linear Regression)' AI 알고리즘은 단기적인 1년의 변화보다 **5년 전체의 거시적인 흐름(평균 기울기)**을 중요하게 평가합니다. 2021~2023년의 하락 폭이 2024년의 반등 폭보다 압도적으로 크기 때문에, 통계적으로 2024년의 상승은 '완전한 추세 전환'이 아닌 '일시적 회복(기술적 반등)'으로 해석되었습니다. 
""")

st.success("""
**💡 사회·경제적 요인 분석 (보수적 예측의 현실적 근거)**

1. **국가적 인구 구조의 변화 (절대 인구 감소):** * 대한민국의 전체 총인구 및 생산가능인구가 본격적으로 감소하고 있습니다. 이는 귀농귀촌을 시도할 수 있는 '잠재 수요층' 자체가 줄어들고 있음을 의미하여 장기적인 우하향의 가장 큰 원인입니다.
2. **높아진 경제적 진입 장벽:** * 고금리 기조가 이어지고, 자재비 및 농지/전원주택 매입 단가가 상승하면서 귀농귀촌에 필요한 '초기 자본 부담'이 과거 2021년 호황기보다 훨씬 높아졌습니다.
3. **농업 불확실성 증가 (귀농 특화):** * 기후 변화(이상 기후)로 인한 농작물 피해 위험성과 농업 소득의 불안정성이 '전업 농업'을 목표로 하는 순수 귀농 결정을 주저하게 만드는 강력한 허들로 작용하고 있습니다.
""")

st.divider()

# 4. 그래프 시각화
st.header("📈 귀농 및 귀촌 연도별 추이 시각화")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 귀농 그래프
ax1.plot(df['연도'], df['귀농인구(명)'], marker='o', color='green', label='과거 데이터')
ax1.plot([2024, 2026], [df['귀농인구(명)'].iloc[-1], pred_farming_2026], marker='o', linestyle='--', color='red', label='2026 AI 예측')
ax1.set_title("귀농 인구 추이 (만 명대)")
ax1.set_xticks([2020, 2021, 2022, 2023, 2024, 2026])
ax1.legend()

# 귀촌 그래프
ax2.plot(df['연도'], df['귀촌인구(명)'], marker='o', color='blue', label='과거 데이터')
ax2.plot([2024, 2026], [df['귀촌인구(명)'].iloc[-1], pred_rural_2026], marker='o', linestyle='--', color='red', label='2026 AI 예측')
ax2.set_title("귀촌 인구 추이 (수십만 명대)")
ax2.set_xticks([2020, 2021, 2022, 2023, 2024, 2026])
ax2.legend()

st.pyplot(fig)
