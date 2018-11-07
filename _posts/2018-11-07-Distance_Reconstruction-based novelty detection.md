---
layout: post
title: Distance/Reconstruction-based novelty detection
author: YH.Kim
---



## Distance/Reconstruction-based novelty detection 
-----

본 게시글에서는 **거리(Distance)와 재구축(Reconstruction) 기반의 이상치 탐지(novelty detection) 기법**들에 대해 설명드리겠습니다.

**본 내용은 고려대 강필성 교수님의 Business-Analytics 수업을 바탕으로    작성하였습니다.*


기법의 소개에 앞서 **이상치 탐지**에 대하여 설명 드리겠습니다.

**이상치 탐지(Novelty Detection)**은 보통 **범주(label)의 심각한 불균형**이 존재하는 상황에서 다수의 범주를 학습하고 **소수 범주를 분류**하는 것입니다.

이상치 데이터(novel data/outliers)에 대한 정의는 조금씩 상이하나,     주된 내용으로는 아래와 같습니다.
- **"수집된 데이터에서 다른 관측치(observation)와   비교하였을 때, 범위에서 많이 벗어난 것"** 


이상치 데이터에 대한 정의들은 아래와 같습니다.

![그림1](D:\dog-k.github.io\_posts\그림1.png)

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

kNN은 가장 기초적인 인스턴스 기반 알고리즘으로, 모든 인스턴스가 n차원의 공간 R^n에 있는 점에 해당한다고 가정합니다[^(T. M. Mitchell, "Machine Learning," 1997)](https://www.cs.ubbcluj.ro/~gabis/ml/ml-books/McGrawHill%20-%20Machine%20Learning%20-Tom%20Mitchell.pdf).
kNN에서 **특정 인스턴스의 범주**는 k개의 최근접한(일반적으로 유클리디언 거리 기반) **이웃 인스턴스들의 주된 범주에 의하여 결정**됩니다.

kNN의 알고리즘은 아래와 같습니다.

![그림2](D:\dog-k.github.io\_posts\그림2.png)

상기의 그림은 kNN이 "파란색 사각형"과 "빨간색 삼각형"으로 이루어진 데이터 집합에서, 초록색 원으로 된 데이터가 주어졌을 때 분류하는 것을 나타낸 것입니다. 그림에서 k를 3으로 두면 실선의 원안에 포함되므로 "빨간색삼각형"으로 분류됩니다. 그러나, k를 5개로 두면 점선의 원안에 포함되므로 "파란색사각형"으로 분류됩니다.

**kNN을 Novelty Detection에 응용한 kNN-Novelty Detection의 주된 내용**은 아래와 같습니다.
- Novelty score of an instance is computed **based on the distance information to k nearest neighbors**
- **Does not assume any prior probability distribution** for the normal class

상기의 내용을 풀어서 설명한 것은 다음과 같습니다.
1. 특정 인스턴의 Novelty score(분류 label)는 k개의 최근접 이웃 인스턴스들의 거리정보(다양한 거리정보 방법 사용)에 기초하여 계산
2. 일반적인 범주에 대한 사전 확률분포를 가정하지 않음 - 인접한 데이터의 범주를 기준으로 분류

Novelty score를 위한 다양한 거리정보 계산은 아래와 같습니다.
![그림3](D:\dog-k.github.io\_posts\그림3.png)

상기의 식들의 전반적인 뜻은 다음과 같습니다.

 ①은 가장 먼거리를 Novelty score로 표현합니다.
 ②는 전체 거리의 평균을 Novelty score로 표현합니다.
 ③는 점들의 mean vector와 인스턴스와의 거리를 Novelty distance로 표현합니다. 이렇게하면 이웃들이 어디에 위치해 있는지 고려할 수 있다는 장점이있습니다.

아래의 그림은 상기의 거리정보 방법들을 그림으로 나타낸 것입니다[^(P. Kang and S. Cho, "A hybrid novelty score and its use in keystroke dynamics-based user authentication," 2009)](https://www.sciencedirect.com/science/article/pii/S0031320309001502).

![그림4](D:\dog-k.github.io\_posts\그림4.png)

