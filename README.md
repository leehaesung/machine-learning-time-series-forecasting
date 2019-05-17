# 시계열 예측을 위해 기계 학습을 사용하는 방법 (하지 않음) : 함정 피하기

* [출처](https://www.kdnuggets.com/2019/05/machine-learning-time-series-forecasting.html)
* [저자:Vegard Flovik](https://www.linkedin.com/in/vegard-flovik/)
* 구글번역사용했음: 오류있음!


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

![이미지1](https://cdn-images-1.medium.com/max/1000/1*38sMNsj2yJCzdQhOfgS87A.jpeg)

이 기사의 모델 구현에 초점을 맞추지 않으므로 모델 정확도를 평가하는 과정을 직접 진행해 보겠습니다. 위의 그림을 육안으로 검사하는 것만으로도 모델 예측은 실제 지표를 따라 가며 정확도가 높습니다. 그러나 좀 더 정확하게하기 위해 아래에 설명 된대로 스 캐터 플롯에 실제 값과 예측 값을 플로팅하고 일반적인 오류 메트릭 [R2 점수](https://en.wikipedia.org/wiki/Coefficient_of_determination)를 계산하여 모델 정확도를 평가할 수 있습니다.

![이미지2](https://cdn-images-1.medium.com/max/1000/1*7nl-TN6iO09mFxBLjQzubA.jpeg)

모델 예측을 통해 R2 점수 0.89를 얻었으며 실제 값과 예측값 사이에 좋은 일치가있는 것으로 보입니다. 그러나 이제 좀 더 자세하게 논의 하겠지만이 메트릭 및 모델 평가는 오해의 소지가 있습니다.

### 이것은 단순히 잘못되었습니다 ....
위의 그림과 계산 된 오류 측정 기준에서 모델은 분명히 정확한 예측을 제공합니다. 그러나 이것은 전혀 사실이 아니며 모델 정확도를 평가할 때 잘못된 정확도 메트릭을 선택하는 것이 매우 잘못된 방법 일 수있는 예일뿐입니다. 이 예에서 설명하기 위해 데이터는 실제로 예측할 수없는 데이터를 나타 내기 위해 명시 적으로 선택되었습니다. 더 구체적으로 말하자면, "주식 인덱스"라고 불리는 데이터는 실제로 무작위 걸음 걸이 프로세스를 사용하여 모델링되었습니다. 이름에서 알 수 있듯이 [랜덤워크](https://en.wikipedia.org/wiki/Random_walk)은 완전히 [확률적인 과정](https://en.wikipedia.org/wiki/Stochastic_process)입니다. 이 때문에 행동 데이터를 학습하고 향후 결과를 예측하기 위해 과거 데이터를 교육용 세트로 사용한다는 생각은 불가능합니다. 이 점을 감안할 때 모델이 어떻게 우리에게 그러한 정확한 예측을주는 것일 수 있습니까? 좀 더 자세하게 돌아가겠습니다. 정확성 메트릭의 선택이 잘못되었습니다.

### 시간 지연 예측 및 자기 상관
이름에서 알 수 있듯이 시계열 데이터는 일시적인 측면이 중요하다는 점에서 다른 유형의 데이터와 다릅니다. 긍정적 인 점은 입력 기능에 유용한 정보뿐만 아니라 시간 경과에 따른 입/출력의 변화가 포함 된 기계 학습 모델을 작성할 때 사용할 수있는 추가 정보를 제공한다는 것입니다. 그러나 시간 구성 요소가 추가 정보를 추가하는 동안 많은 다른 예측 작업과 비교하여 시계열 문제를 처리하기가 더 어려워집니다.

이 특정 예에서는 이전에 데이터에 따라 예측을 수행 하는 [LSTM 네트워크](https://en.wikipedia.org/wiki/Long_short-term_memory)를 사용했습니다  . 그러나 아래 그림과 같이 모형 예측을 약간 확대하면 모형이 실제로 무엇을하는지 알기 시작합니다.

![이미지3](https://cdn-images-1.medium.com/max/1000/1*A-ubY-due4lcTEOgGSCvoQ.jpeg)

시계열 데이터는 시간에 상관되는 경향이 있으며 중요한 [자기상관관계](https://en.wikipedia.org/wiki/Autocorrelation)를 나타냅니다. 이 경우, 이는 시간 "t + 1" 에서의 인덱스가 시간 "t" 에서 인덱스에 매우 근접 할 가능성이 있음을 의미합니다. 위의 그림에서 오른쪽 그림에서와 같이 모델이 실제로 수행하는 작업은 시간 "t + 1" 에서 값을 예측할 때 단순히 시간 "t" 의 값을 예측으로 사용한다는 것입니다(종종  지속성 모델). 는 C 플로팅 [로스 상관](https://en.wikipedia.org/wiki/Cross-correlation) 예측과 실제 값 사이를 (아래 그림), 우리는 모델이 단순히 미래에 대한 예측과 이전 값을 사용하는 것을 나타내는, 일일의 시간 지연에 명확한 피크를 참조하십시오.

![이미지4](https://cdn-images-1.medium.com/max/1000/1*4_LdgQFVtI8dHKYKYvFpNA.jpeg)

### 부정확하게 사용하면 정확도 메트릭이 오도 될 수 있습니다.
즉, 직접적으로 값을 예측할 수있는 능력을 기준으로 모델을 평가할 때 [평균 백분율 오차](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error) 및 [R2 점수](https://en.wikipedia.org/wiki/Coefficient_of_determination)와 같은 공통적 인 오차 척도는 모두 높은 예측 정확도를 나타냅니다. 그러나 예제 데이터는 무작위 도보 과정을 통해 생성되므로 모델은 향후 결과를 예측할 수 없습니다. 이는 일반적인 오류 메트릭스를 직접 계산하여 예측력을 모델링하는 것만으로도 오도 된 결과를 낳을 수 있다는 사실과 모델 정확성에 지나치게 자신감을 갖도록 속일 수 있다는 사실을 강조합니다.

### 연속성 및 차분 시계열 데이터
[고정 시계열](https://www.otexts.org/fpp/8/1) 그의 통계적 성질 등이다. [평균](https://en.wikipedia.org/wiki/Mean), [분산](https://en.wikipedia.org/wiki/Variance), [자기 상관](https://en.wikipedia.org/wiki/Autocorrelation) 등, 시간 경과에 모두 일정하다. 대부분의 통계 예측 방법은 시계열이 수학적 변형을 사용하여 거의 정지 상태 (즉, "고정 된 상태")로 렌더링 될 수 있다는 가정을 기반으로합니다. 이러한 기본 변환 중 하나는 아래그림과 같이 [데이터의 시차를 설정하는 것](https://www.otexts.org/fpp/8/1) 입니다.

![이미지5](https://cdn-images-1.medium.com/max/1000/1*UzcA8R7Y8Evcwd5uRwuqEw.jpeg)

이 변환은 색인을 직접 고려하기보다는 연속 된 시간 간격 의 차이  를 계산  합니다.

가치 그 자체보다는 시간 단계 간의 가치 의 차이 를 예측하기위한 모델을 정의하는  것은 모델의 예측력을 훨씬 더 강력하게 테스트하는 것입니다. 이 경우 단순히 데이터에 강력한 자기 상관성이 있음을 단순히 사용할 수 없으며 "t + 1"에 대한 예측으로 시간 "t" 의 값을 사용합니다 . 이로 인해 모델에 대한 더 나은 테스트를 제공하고 교육 단계에서 유용한 것을 배웠는지, 그리고 과거 데이터를 분석하면 실제로 모델이 미래의 변화를 예측하는 데 도움이되는지 여부가 파악됩니다.

### 시간 차분 데이터의 예측 모델
데이터가 아닌 시간 차분 데이터를 직접 예측할 수 있기 때문에 모델의 예측 능력을 훨씬 더 강력하게 나타낼 수 있으므로 모델을 사용해 보겠습니다. 이 테스트의 결과는 아래 그림과 같이 실제 값과 예측 값의 분산 플롯을 보여줍니다.

![](https://cdn-images-1.medium.com/max/1000/1*HeXMVb737QXHH1smBx3MUg.jpeg)

이 수치는  데이터가 완전히 확률 론적 [랜덤워크 걸이 프로세스](https://en.wikipedia.org/wiki/Random_walk)를 사용하여 생성되기 때문에 모델이 과거 사건을 기반으로 미래의 변화를 예측할 수  없다는 것을 나타냅니다  . 확률 론적 과정 의 미래 결과를 예측할 수 있다는 것은   정의상 불가능하며, 누군가가 이것을 주장한다면, 조금 회의적이어야합니다 ...

### 당신의 시계열은 무작위 도보입니까?
당신의 시계열은 실제로 무작위 도보 일 수 있으며이를 확인하는 몇 가지 방법은 다음과 같습니다.
* 시계열은 선형 또는 유사한 패턴으로 쇠퇴하는 강한 시간 종속성 (자기 상관)을 보여줍니다.
* 시계열은 고정적이지 않고 고정되어있어 데이터에서 명백하게 학습 가능한 구조를 나타내지 않습니다.
* 지속성 모델 (이전 시간 단계에서의 관찰을 다음 시간 단계에서 발생하는 것으로 사용)은 신뢰할 수있는 예측의 최상의 원천을 제공합니다.

이 마지막 지점은 시계열 예측의 핵심입니다. [퍼시스턴스 모델](https://machinelearningmastery.com/persistence-time-series-forecasting-with-python/)을 사용한 기본 예측은 빠르게 향상시킬 수 있는지 여부를 나타냅니다. 할 수 없다면 아마 무작위 도보 (또는 그 근처)를 다룰 것입니다. 인간의 마음은 모든 곳에서 패턴을 찾기 위해 고정되어 있으며 우리는 무작위로 걷는 과정을 위해 정교한 모델을 개발함으로써 우리 자신을 속이고 시간을 낭비하지 않도록 경계해야합니다.

저자 참고 사항 : 기사를 게시 한 후 Jason Brownlee의 무작위 산책 및 시간 시리즈와 유사한 주제의 멋진 기사를 알게되었습니다 . [Python으로 Times Series Forecasting을위한 랜덤워크.](https://machinelearningmastery.com/gentle-introduction-random-walk-times-series-forecasting-python/)

### 개요
이 기사를 통해 강조하고자하는 요점   은 예측 정확도 측면에서 모델 성능을 평가할 때 매우 신중 해야한다는 것입니다. 위의 예에서 볼 수 있듯이, 미래의 결과를 예측하는 것이 완전히 불가능한 완전히 무작위 인 프로세스 일지라도, 쉽게 속일 수 있습니다. 단순히 모델을 정의하고 예측을 수행하고 공통 정확도 메트릭을 계산하면 겉으로는 좋은 모델을 가지고이를 생산에 적용 할 수 있습니다. 반면에 실제로 모델에는 예측력이 전혀 없을 수도 있습니다.

당신이 시계열 예측과 함께 일하고 있고, 아마도 당신 자신을 데이터 과학자라고 생각한다면, 나는 [과학자](https://en.wikipedia.org/wiki/Scientist) 측면에 대해서도 강조 할 것을 촉구합니다. 항상 데이터에 대해 회의적이며 비판적인 질문을하고 결코 성급한 결론을 내릴 수 없습니다. [과학적인 방법은](https://en.wikipedia.org/wiki/Scientific_method)  과학의 다른 종류로 데이터 과학에 적용되어야한다.

[원본](https://towardsdatascience.com/how-not-to-use-machine-learning-for-time-series-forecasting-avoiding-the-pitfalls-19f9d7adf424). 허락을 받아 재 게시.

Bio :  [Vegard Flovik](https://www.linkedin.com/in/vegard-flovik/) 은 리드 데이터 과학자입니다. Axbit AS의 기계 학습 및 고급 분석.


자원:
* [온라인 및 웹 기반 : 분석, 데이터 마이닝, 데이터 과학, 기계 학습 교육](https://www.kdnuggets.com/education/online.html)
* [분석, 데이터 과학, 데이터 마이닝 및 기계 학습을위한 소프트웨어](https://www.kdnuggets.com/software/index.html)

관련 항목 :
* [간단한 신경 회로망 및 LSTM을 이용한 시계열 예측 소개](https://www.kdnuggets.com/2019/04/introduction-time-series-forecasting-simple-neural-networks-lstm.html)
* [자동화 된 기계 학습을 통한 시계열 분석 가속화](https://www.kdnuggets.com/2019/02/datarobot-accelerating-time-series-analysis-automated-machine-learning.html)
* [예측 정확도를 향상시키기 위해 모델 학습을 미세 조정하는 방법](https://www.kdnuggets.com/2019/01/fine-tune-machine-learning-models-forecasting.html)

