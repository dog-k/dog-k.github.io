---
layout: post
title: Distance/Reconstruction-based novelty detection
author: YH.Kim
---




-----

본 게시글에서는 **거리(Distance)와 재구축(Reconstruction) 기반의 이상치 탐지(novelty detection) 기법**들에 대해 설명드리겠습니다.

**본 내용은 고려대 강필성 교수님의 Business-Analytics 수업을 바탕으로    작성하였습니다.*


기법의 소개에 앞서 **이상치 탐지**에 대하여 설명 드리겠습니다.

**이상치 탐지(Novelty Detection)**은 보통 **범주(label)의 심각한 불균형**이 존재하는 상황에서 다수의 범주를 학습하고 **소수 범주를 분류**하는 것입니다.

이상치 데이터(novel data/outliers)에 대한 정의는 조금씩 상이하나,     주된 내용으로는 아래와 같습니다.
- **"수집된 데이터에서 다른 관측치(observation)와   비교하였을 때, 범위에서 많이 벗어난 것"** 


이상치 데이터에 대한 정의들은 아래와 같습니다.

![그림1](https://i.imgur.com/4ZghtiF.png)

상기와 같은 내용을 보았을 때 분석에서 random error인 Nosie data와 혼동이 있을 수 있을수 있으나, **이상치와 노이즈는 상이**합니다. 그 내용은 아래와 같습니다.
- Noise is random error or variance in a measured variable
- Noise should be removed before outlier dectection



본 게시글에서는 Novelty detection의 3가지 알고리즘을 소개합니다.
1. k-Nearest Neighbor-based Novelty Detection
2. K-Means clustering-based Novelty Detection
3. Principal Component Analysis-based Novelty Detection


-----
## k-Nearest Neighbor-based Novelty Detection

첫번째로 소개해드릴 기법은 k-Nearest Neighbor-based Novelty Detection(이하 kNN-Novelty Detection)입니다.

본 기법은 잘알려진 인스턴스 기반(instance-based) 또는 lazy 학습 분류 알고리즘인 k-최근접이웃(k-Nearest Neighbor, kNN)를 응용한 것입니다. 따라서 kNN-Novelty Detection을 설명드리기전에 kNN을 먼저 설명드리겠습니다.

kNN은 가장 기초적인 인스턴스 기반 알고리즘으로, 모든 인스턴스가 n차원의 공간 R^n에 있는 점에 해당한다고 가정합니다[T. M. Mitchell, "Machine Learning," 1997](https://www.cs.ubbcluj.ro/~gabis/ml/ml-books/McGrawHill%20-%20Machine%20Learning%20-Tom%20Mitchell.pdf).
kNN에서 **특정 인스턴스의 범주**는 k개의 최근접한(일반적으로 유클리디언 거리 기반) **이웃 인스턴스들의 주된 범주에 의하여 결정**됩니다.

kNN의 알고리즘은 아래와 같습니다.

![그림2](https://i.imgur.com/Kt59PjK.png)

상기의 그림은 kNN이 "파란색 사각형"과 "빨간색삼각형"으로 이루어진 데이터 집합에서, 초록색 원으로 된 데이터가 주어졌을 때 분류하는 것을 나타낸 것입니다. 그림에서 k를 3으로 두면 실선의 원안에 포함되므로 "빨간색삼각형"으로 분류됩니다. 그러나, k를 5개로 두면 점선의 원안에 포함되므로 "파란색사각형"으로 분류됩니다.

**kNN을 Novelty Detection에 응용한 kNN-Novelty Detection의 주된 내용**은 아래와 같습니다.
- Novelty score of an instance is computed **based on the distance information to k nearest neighbors**
- **Does not assume any prior probability distribution** for the normal class

상기의 내용을 풀어서 설명한 것은 다음과 같습니다.
1. 특정 인스턴스의 Novelty score는 k개의 최근접 이웃 인스턴스들의 거리정보(다양한 거리정보 방법 사용)에 기초하여 계산
2. 일반적인 범주에 대한 사전 확률분포를 가정하지 않음 - 인접한 데이터의 범주를 기준으로 분류

Novelty score를 위한 다양한 거리정보 계산은 아래와 같습니다.
![그림3](https://i.imgur.com/M4xvd9p.png)

상기의 식들의 전반적인 뜻은 다음과 같습니다.

①은 가장 먼거리를 Novelty score로 표현합니다.

②는 전체 거리의 평균을 Novelty score로 표현합니다.

③는 점들의 mean vector와 인스턴스와의 거리를 Novelty distance로 표현합니다. 이렇게하면 이웃들이 어디에 위치해 있는지 고려할 수 있다는 장점이있습니다.

아래의 그림은 상기의 거리정보 방법들을 그림으로 나타낸 것입니다[(P. Kang and S. Cho, "A hybrid novelty score and its use in keystroke dynamics-based user authentication," 2009)](https://www.sciencedirect.com/science/article/pii/S0031320309001502).

![그림4](https://i.imgur.com/DadMPTe.png)

아래의 그림은 상기에서 설명한 식들을 이용하여 주어진 데이터에 novelty score를 적용한 것입니다.

![그림5](https://i.imgur.com/RCSOgIC.png)
출처: https://github.com/pilsung-kang/Business-Analytics

상기의 그림에서 A와 B 모두에서 직관적으로 "빨간색원"이 이상치로 보이나 거리정보를 이용하였을 때는 "빨간색삼각형"을 이상치로 더 많이 분류하였습니다.
위와 같은 문제점을 해결하기 위하여 추가적인 거리정보를 이용하며 그 식은 아래와 같습니다.

![그림6](https://i.imgur.com/gegsqRY.png)

상기의 식들의 전반적인 뜻은 다음과 같습니다.

①은 전체 거리의 평균을 Novelty score로 표현합니다. 이는 앞서본 식과 동일합니다.

②는 이웃들 내에 위치하면 Novelty score를 0으로 밖에 위치하면 패널티를 부여하는 방식으로 자세하게는 아래 그림과 같습니다.

![그림7](https://i.imgur.com/kk50bgk.png)
출처: https://github.com/pilsung-kang/Business-Analytics

③은 ①과 ②의 식을 조합한 것입니다.

아래의 그림은 상기의 거리정보들의 이상치 분류 결과를 나타냅니다.

![그림8](https://i.imgur.com/HFlOfhD.png)
출처: https://github.com/pilsung-kang/Business-Analytics

상기의 그림에서 추가적인 거리정보를 모두 조합한 hybrid식이 이상치를 잘분류해 내는 것을 알 수 있습니다.



-----
## Clustering-based Novelty Detection

두번째로 소개해드릴 기법은 Clustering-based Novelty Detection이며, 그 중에서도 K-Means clustering-based Novelty Detection(K-Means clustering-Novelty Detection)입니다.

본 기법은 주어진 데이터를 k개의 군집으로 묶는 군집화 알고리즘인 K-평균 군집화(K-Means clustering)를 응용한 것입니다. 따라서 이번에도 K-평균 군집화를 먼저 설명드리겠습니다.

군집화(Clustering)는 주어진 데이터에서 예측할 범주는 없지만 인스턴스들이 각 성격에 맞는 그룹들로 분할될 경우 적용할 수 있는 방법입니다. 그 중 **K-평균 군집화는 반복적인 거리 기반**으로 기초적인 군집화 알고리즘입니다[(I. H. Witten, E. Frank and M. A. Hall, "Data Mining: Practical Machine Learning Tools and Techniques," 2011)](https://dl.acm.org/citation.cfm?id=1972514).

K-평균 군집화는 **K개의 군집수를 정하고 군집의 중심이 될 K개의 점을 데이터 중에서 임의**로 선택합니다. 일반적인 유클리드 거리를 이용하여 모든 인스턴스들을 각각 자신들에게 **가장 가까운 군집에 할당하고 각 군집에 속한 인스턴스들의 평균**을 계산합니다. 다음으로 구한 평균값을 **군집의 새로운 평균값으로 사용하고 평균값이 더이상 변화가 없을 때까지 반복**합니다. 아래의 그림은 K-평균 군집화 실행과정을 그림으로 나타낸 것입니다.

![그림9](https://i.imgur.com/vy8zWNE.png) 출처: https://ko.wikipedia.org/

상기에서 **K값은 분석자가 임의로 지정하는 하이퍼파라미터**이나, 실루엣너비, GAP 통계량과 같은 **최적군집수 결정 방법을 사용**하여 정하기도합니다. 최적군집수 결정 방법에 대한 [링크](http://statdb1.uos.ac.kr/teaching/multi-under/cluster-number.pdf)입니다. 궁금하신분들은 참고하시면 좋을것 같습니다.

**K-평균 군집화를 Novelty Detection에 응용한 K-Means clustering-Novelty Detection의 주된 내용**은 아래와 같습니다.
- Novelty score of an instance is computed **based on the distance information to the nearest centroid**
- **Does not assume any prior probability distribution** for the normal class


![그림10](https://i.imgur.com/xQFGjxh.png)


상기의 그림에서 파란색과 주황색 원이 해당 군집까지의 거리가 동일하다고 할 때, 우리는 직관적으로 주황색 원이 이상치라는 것을 알 수 있습니다. **K-Means clustering-Novelty Detection은 가장 가까운 중심과의 절대거리, 군집의 지름을 이용하여 Novelty score**를 계산합니다. 식은 아래와 같습니다.


![그림11](https://i.imgur.com/rTtyKYz.png)




-----
## Principal Component Analysis-based Novelty Detection

마지막으로 소개해드릴 기법은 Principal Component Analysis-based Novelty Detection(이하 PCA-Novelty Detection) 입니다.

본 기법은 주어진 데이터의 분산이 최대화 되도록 차원을 축소하는 방법인 주성분분석(Principal COmponent Analysis, PCA)를 응용한 것입니다. 마찬가지로 PCA를 먼저 설명드리겠습니다.

**PCA는 고차원의 데이터를 저차원의 데이터로 환원**시키는 기법으로 **주어진 데이터의 분산이 최대가 되는 축을 1주성분**으로 그 다음 분산이 최대가 되는 축을 2주성분으로 구성하는 방식으로 **원데이터(raw data)의 차원과 동일하거나 보다 작게** 만듭니다[A. Geron, "Hands-On Machine Learning with Scikit-Learn and Tensorflow," 2017](http://shop.oreilly.com/product/0636920052289.do).

아래의 그림은 PCA로 데이터의 주성분을 어떻게 찾는지를 나타낸 것입니다.
![그림12](https://i.imgur.com/T8GO4GU.png) 출처: https://github.com/ageron/handson-ml

상기의 그림에서 보는 것과 같이 PCA는 주어진 데이터의 차원을 축소하나, 분산(변동성)을 가장 잘 보존하는 변수(주성분)를 찾아내는 기법입니다. PCA는 원데이터의 변수를 선형결합(Linear Combination)하여 새로운 변수(주성분)를 만들어 내는데, 이때 데이터들을 주성분 축에 사영(Projection) 시킵니다.

PCA의 보다 세부적인 설명과 방법은 다음의 [링크](https://ratsgo.github.io/machine%20learning/2017/04/24/PCA/)를 참고하시면 될 듯합니다.

추가적으로 PCA를 사용하여 차원을 축소할 때, 어느정도까지의 차원 축소가 적당한지에 대한 것은 **분석자가 설정하기에 따라 다르나 보통의 경우 Elobow plot**을 이용하거나 분산(변동성)이 70%이상일 때를 기준으로 사용합니다. Elbow plot의 예시는 아래와 같습니다.

![그림13](https://i.imgur.com/pDEKarM.png) 출처: https://wikidocs.net/7646



**PCA를 Novelty Detection에 응용한 PCA-Novelty Detection의 주된 내용**은 아래와 같습니다.
- Novelty score: the **amount of reconstruction loss from the projected space into the original space**

상기의 주된 내용으로는 원데이터를 사영시킨 공간과 원데이터의 공간 사이의 에러가 클수록 이상치 점수를 주는 것입니다.

아래의 그림은 원데이터에 PCA후 다시 복원시키는 과정을 나타낸 것입니다.

![그림14](https://i.imgur.com/6cyKYy0.png)
출처: https://github.com/pilsung-kang/Business-Analytics

상기의 그림에서 X는 원데이터이며 w^T를 이용하여 저차원으로 사영시키고, 다시 w를 곱하여 원래 차원으로 사영시키는 것을 나타냅니다.

아래의 그림은 PCA-Novelty Detection의 방법과 그림을 나타낸 것입니다.
![그림15](https://i.imgur.com/ZABmgo8.png)
출처: https://github.com/pilsung-kang/Business-Analytics

