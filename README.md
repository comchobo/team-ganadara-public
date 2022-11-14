# 2022 국립국어원 인공 지능 언어 능력 평가 
# team-ganadara
packages below are needed :
- torch==1.12.1
- ㄴ install by `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`
- `pip install transformers==4.22.2 numpy==1.23.3 datasets==2.5.2 scikit-learn jsonlines optuna`

# 사용 방법
- ABSA baseline 모델 구조 차용
- LMkor_Electra Pretrained 모델 사용
- (핵심 아이디어) ABSA baseline의 레이블 불균형을 해결하기 위하여 Class-weight 기법 사용
- 7 Ensemble 사용

# 참여 인원
- 지승현 (sorryhyun@soongsil.ac.kr)
- 황현빈 (hhb9817@naver.com)
