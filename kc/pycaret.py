# PyCaret 설치
# !pip install pycaret

# https://stackoverflow.com/questions/73460897/modulenotfounderror-no-module-named-pycaret-time-series
# 아래 설치해야됨
# pip install pycaret-ts-alpha

# 데이터 불러오기
from pycaret import *
# from pycaret import show_versions

# print(pycaret.show_versions())

from pycaret.datasets import get_data
data = get_data('iris')

# 분류 모델 생성
from pycaret.classification import *
clf = setup(data=data, target='species')

# 모델 비교
compare_models()

# 최적의 모델 선택 및 학습
best_model = create_model('rf')

# 모델 평가
evaluate_model(best_model)

# 모델 예측
predict_model(best_model)

# 모델 저장
save_model(best_model, 'model_name')
