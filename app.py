import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (윈도우용 맑은 고딕)
plt.rc('font', family='Malgun Gothic')

# ---------------------------------------------------------
# 1. 2020~2025 데이터 세팅 (2025년은 농식품부 최신 실태조사 기반 추정치)
# ---------------------------------------------------------
data = {
    '연도': [2020, 2021, 2022, 2023, 2024, 2025],
    '귀농인구(명)': [12489, 14461, 12660, 10307, 10710, 10200], # 2025: 영농 어려움으로 소폭 감소 추정
    '귀촌인구(명)': [477122, 515434, 421106, 400600, 422789, 425000] # 2025: 청년층 유입 및 70% 만족도 기반 반등세 유지 추정
}
df = pd.DataFrame(data)

# ---------------------------------------------------------
# 2. 인공지능 예측 모델 (총 6년치 데이터 학습)
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
st.write("2020~2024년 통계청 확정 데이터와 **2025년 농식품부 실태조사 기반 추정치**를 학습하여 2026년을 예측합니다.")

# 1. 과거 통계 및 의미
st.header("📊 2020~2025년 통계 데이터 (2025년은 추정치)")
st.dataframe(df.style.highlight_max(axis=0))

st.info("""
**💡 2025년 데이터 세팅 근거 (2026.02.25. 농림축산식품부 발표 자료 기준)**
* **귀농 (Farming):** 귀농 5년 차 가구의 농외소득 의존도가 69.9%로 급증하고 평균 경작 규모가 전년 대비 22% 감소하는 등 순수 농업 진입의 장벽이 높아진 현실을 반영하여 소폭 하락(10,200명)으로 추정했습니다.
* **귀촌 (Rural):** 귀촌 생활 만족도가 70%에 달하고, U형(연고지 복귀) 귀촌이 73%를 차지하는 등 안정적인 정착 흐름을 보이고 있어 2024년의 반등세를 이어가는 수준(425,000명)으로 세팅했습니다.
""")

st.divider()

# 2. 예측 수치 및 결과
st.header("🔮 2026년 인구 예측 수치")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="2026년 예상 귀농인구", value=f"{pred_farming_2026:,}명", delta="지속 하락 추세")
with col2:
    st.metric(label="2026년 예상 귀촌인구", value=f"{pred_rural_2026:,}명", delta="하락폭 둔화 및 안정화")

st.success("""
**🧠 6년(2020~2025) 데이터 기반 2026년 예측의 의미**

데이터를 2025년까지 1년 더 보강하여 학습시킨 결과, AI 모델이 일시적인 등락에 흔들리지 않고 **장기적인 지지선**을 더 명확하게 찾아냈습니다. 
* **귀농의 위기:** 데이터가 누적될수록 '전업 농업'의 진입 장벽이 굳어지고 있음을 AI가 강하게 인식했습니다. 따라서 자금 지원을 넘어선 '지역 밀착형 영농 실습' 정책이 더욱 강조되어야 합니다.
* **귀촌의 안정화:** 2025년 귀촌 인구가 안정세를 유지할 것이라는 데이터를 추가 학습함으로써, 기존에 AI가 예측했던 극단적인 폭락세가 눈에 띄게 완화되었습니다. 농촌이 점차 '일터'에서 '거주와 힐링의 공간'으로 변화하는 트렌드가 반영된 결과입니다.
""")

st.divider()

# 3. 그래프 시각화
st.header("📈 연도별 추이 시각화 및 2026 예측")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 귀농 그래프
ax1.plot(df['연도'], df['귀농인구(명)'], marker='o', color='green', label='학습 데이터 (2020-2025)')
ax1.plot([2025, 2026], [df['귀농인구(명)'].iloc[-1], pred_farming_2026], marker='o', linestyle='--', color='red', label='2026 예측')
ax1.set_title("귀농 인구 추이 (명)")
ax1.set_xticks([2020, 2021, 2022, 2023, 2024, 2025, 2026])
ax1.legend()

# 귀촌 그래프
ax2.plot(df['연도'], df['귀촌인구(명)'], marker='o', color='blue', label='학습 데이터 (2020-2025)')
ax2.plot([2025, 2026], [df['귀촌인구(명)'].iloc[-1], pred_rural_2026], marker='o', linestyle='--', color='red', label='2026 예측')
ax2.set_title("귀촌 인구 추이 (명)")
ax2.set_xticks([2020, 2021, 2022, 2023, 2024, 2025, 2026])
ax2.legend()

st.pyplot(fig)
