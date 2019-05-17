# 시계열 예측을 위해 기계 학습을 사용하는 방법 (하지 않음) : 함정 피하기

* [출처](https://www.kdnuggets.com/2019/05/machine-learning-time-series-forecasting.html)
* [저자:Vegard Flovik](https://www.linkedin.com/in/vegard-flovik/)



우리는 시간 지연 예측, 자기 상관, 확률, 정확도 메트릭 등을 보면서 시계열 예측에 대한 기계 학습의 일반적인 함정을 간략하게 설명합니다.

내 다른 게시물에서 나는 같은 주제를 커버한  [기계 학습 및 물리 결합하는 방법](https://towardsdatascience.com/how-do-you-combine-machine-learning-and-physics-based-modeling-3a3545d58ab9) , 방법과  [기계 학습 생산 최적화를 위해 사용할 수 있습니다.](https://towardsdatascience.com/machine-learning-for-production-optimization-e460a0b82237)  뿐만 아니라,  이상 탐지 및 상태 모니터링을 . 그러나이 글에서는 시계열 예측을위한 기계 학습의 공통적 인 함정에 대해 논의 할 것입니다.

시계열 예측은 기계 학습의 중요한 영역입니다. 시간 구성 요소와 관련된 많은 예측 문제가 있기 때문에 중요합니다. 그러나 시간 구성 요소가 추가 정보를 추가하는 동안 많은 다른 예측 작업과 비교하여 시계열 문제를 처리하기가 더 어려워집니다.

이 게시물은  기계 학습을 사용하여 [시계열 예측](https://en.wikipedia.org/wiki/Time_series) 작업을 수행  하고 일반적인 함정 중 일부를 피하는 방법을 설명합니다. 구체적인 예를 통해 필자는 겉으로보기에는 좋은 모델을 어떻게 제작하여 생산에 적용 할 수 있는지를 보여줄 것입니다. 실제로 모델에는 예측 능력이 전혀 없을 수도 있습니다. 구체적으로 모델 정확성을 평가하는 방법에 중점을 둡니다. [평균 백분율 오차](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error) ,  [R2 점수](https://en.wikipedia.org/wiki/Coefficient_of_determination)  등의 일반적인 오류 측정 기준에 단순히 의존하는 것이  신중하게 적용되는 경우 오해의 소지가 있음을 보여줍니다.

### 시계열 예측을위한 기계 학습 모델
시계열 예측에 사용할 수있는 모델에는 여러 가지 유형이 있습니다. 이 특정 예에서는 Long 단기 메모리 네트워크를 사용하거나 짧은 시간 내에  [LSTM 네트워크](https://en.wikipedia.org/wiki/Long_short-term_memory) 를 사용했습니다.이 네트워크는 이전 시간의 데이터에 따라 예측을 수행하는 특수한 종류의 신경 네트워크입니다. 언어 인식, 시계열 분석 및 기타 많은 분야에서 널리 사용됩니다. 그러나 제 경험에 따르면 단순한 유형의 모델은 실제로 많은 경우 정확한 예측을 제공합니다. [랜덤 포레스트](https://en.wikipedia.org/wiki/Random_forest) ,  [그레디언트 부스트 회귀 변수](https://en.wikipedia.org/wiki/Gradient_boosting) 및 [시간 지연 신경망](https://en.wikipedia.org/wiki/Time_delay_neural_network) 과  같은 모델 사용 , 시간 정보는 입력에 추가되는 지연 집합을 통해 포함될 수 있으므로 데이터는 다른 시점에서 나타납니다. TDNN은 순차적 인 특성으로 인해 [반복적 인 신경망](https://en.wikipedia.org/wiki/Recurrent_neural_network)  대신  [피드 포워드 신경망으로](https://en.wikipedia.org/wiki/Feedforward_neural_network) 구현됩니다  .

