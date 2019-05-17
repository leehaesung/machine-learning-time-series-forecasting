# 시계열 예측을 위해 기계 학습을 사용하는 방법 (하지 않음) : 함정 피하기

* [출처](https://www.kdnuggets.com/2019/05/machine-learning-time-series-forecasting.html)
* [저자:Vegard Flovik](https://www.linkedin.com/in/vegard-flovik/)



우리는 시간 지연 예측, 자기 상관, 확률, 정확도 메트릭 등을 보면서 시계열 예측에 대한 기계 학습의 일반적인 함정을 간략하게 설명합니다.

내 다른 게시물에서 나는 같은 주제를 커버한  [기계 학습 및 물리 결합하는 방법](https://towardsdatascience.com/how-do-you-combine-machine-learning-and-physics-based-modeling-3a3545d58ab9) , 방법과  [기계 학습 생산 최적화를 위해 사용할 수 있습니다.](https://towardsdatascience.com/machine-learning-for-production-optimization-e460a0b82237)  뿐만 아니라,  이상 탐지 및 상태 모니터링을 . 그러나이 글에서는 시계열 예측을위한 기계 학습의 공통적 인 함정에 대해 논의 할 것입니다.

시계열 예측은 기계 학습의 중요한 영역입니다. 시간 구성 요소와 관련된 많은 예측 문제가 있기 때문에 중요합니다. 그러나 시간 구성 요소가 추가 정보를 추가하는 동안 많은 다른 예측 작업과 비교하여 시계열 문제를 처리하기가 더 어려워집니다.

이 게시물은  기계 학습을 사용하여 [시계열 예측](https://en.wikipedia.org/wiki/Time_series) 작업을 수행  하고 일반적인 함정 중 일부를 피하는 방법을 설명합니다. 구체적인 예를 통해 필자는 겉으로보기에는 좋은 모델을 어떻게 제작하여 생산에 적용 할 수 있는지를 보여줄 것입니다. 실제로 모델에는 예측 능력이 전혀 없을 수도 있습니다. 구체적으로 모델 정확성을 평가하는 방법에 중점을 둡니다. [평균 백분율 오차](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error) ,  [R2 점수](https://en.wikipedia.org/wiki/Coefficient_of_determination)  등의 일반적인 오류 측정 기준에 단순히 의존하는 것이  신중하게 적용되는 경우 오해의 소지가 있음을 보여줍니다.

### 시계열 예측을위한 기계 학습 모델
시계열 예측에 사용할 수있는 모델에는 여러 가지 유형이 있습니다. 이 특정 예에서는 Long 단기 메모리 네트워크를 사용하거나 짧은 시간 내에  [LSTM 네트워크](https://en.wikipedia.org/wiki/Long_short-term_memory) 를 사용했습니다.이 네트워크는 이전 시간의 데이터에 따라 예측을 수행하는 특수한 종류의 신경 네트워크입니다. 언어 인식, 시계열 분석 및 기타 많은 분야에서 널리 사용됩니다. 그러나 제 경험에 따르면 단순한 유형의 모델은 실제로 많은 경우 정확한 예측을 제공합니다. [랜덤 포레스트](https://en.wikipedia.org/wiki/Random_forest) ,  [그레디언트 부스트 회귀 변수](https://en.wikipedia.org/wiki/Gradient_boosting) 및 [시간 지연 신경망](https://en.wikipedia.org/wiki/Time_delay_neural_network) 과  같은 모델 사용 , 시간 정보는 입력에 추가되는 지연 집합을 통해 포함될 수 있으므로 데이터는 다른 시점에서 나타납니다. TDNN은 순차적 인 특성으로 인해 [반복적 인 신경망](https://en.wikipedia.org/wiki/Recurrent_neural_network)  대신  [피드 포워드 신경망으로](https://en.wikipedia.org/wiki/Feedforward_neural_network) 구현됩니다  .

### 오픈 소스 소프트웨어 라이브러리를 사용하여 모델을 구현하는 방법
나는 일반적으로 사용하는 모델의 내 신경 네트워크 유형을 정의  Keras 높은 수준의 신경 네트워크 API입니다, 파이썬으로 작성된와의 상단에 실행할 수 [TensorFlow](https://github.com/tensorflow/tensorflow),  [CNTK](https://github.com/Microsoft/cntk) , 또는  [Theano](https://github.com/Theano/Theano)을 모델의 다른 유형의 난 보통 사용 [Scikit-Learn](http://scikit-learn.org/stable/) 무료 소프트웨어 기계학습 라이브러리인, 다양한 [통계적 분류](https://en.wikipedia.org/wiki/Statistical_classification), [회귀분석](https://en.wikipedia.org/wiki/Regression_analysis) 및 [클러스터링](클러스터링)을 포함하여 알고리즘 [서포트벡터머신](https://en.wikipedia.org/wiki/Support_vector_machine), [랜덤포레스트](https://en.wikipedia.org/wiki/Random_forests), [그레디언트 부스팅](https://en.wikipedia.org/wiki/Gradient_boosting),  [K-means](https://en.wikipedia.org/wiki/K-means_clustering) 및 [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)을 Python 숫자 및 과학 라이브러리 인 [NumPy](https://en.wikipedia.org/wiki/NumPy) 및 [SciPy](https://en.wikipedia.org/wiki/SciPy) 와 상호 작용하도록 설계되었습니다.

그러나이 기사의 주요 주제는 시계열 예측 모델을 구현하는 방법이 아니라 모델 예측을 평가하는 방법입니다. 이 때문에, 나는 다른 블로그 게시물과 그 주제를 다루는 기사가 많기 때문에 모형 제작 등의 세부 사항을 다루지 않을 것이다.

### 사례 : 시계열 데이터 예측
이 경우 사용 된 예제 데이터는 아래 그림에 나와 있습니다. 나중에 더 자세히 데이터로 돌아갈 것이지만, 지금은이 데이터가 주식 지수의 연례 진화를 나타내는 것으로 가정하겠습니다. 데이터는 첫 번째 250 일이 모델의 학습 데이터로 사용되는 교육 및 테스트 세트로 분할되고 데이터 세트의 마지막 부분에서 주가 지수를 예측하려고 시도합니다.

![이미지](https://cdn-images-1.medium.com/max/1000/1*38sMNsj2yJCzdQhOfgS87A.jpeg)
