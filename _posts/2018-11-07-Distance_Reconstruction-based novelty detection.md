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

상기에서 말하는 Convex Hull distance는 아래의 그림과 같습니다.
![그림7.1](https://i.imgur.com/9pCvoQ4.png)

아래의 그림은 상기의 거리정보들의 이상치 분류 결과를 나타냅니다.

![그림8](https://i.imgur.com/HFlOfhD.png)
출처: https://github.com/pilsung-kang/Business-Analytics

상기의 그림에서 추가적인 거리정보를 모두 조합한 hybrid식이 이상치를 잘분류해 내는 것을 알 수 있습니다.

아래는 kNN-Novelty Detection의 파이썬 코드를 나타낸 것입니다.

```
import numpy as np
import math
import operator
 

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1

    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
  
    distances.sort(key=operator.itemgetter(1))
    
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
 
    return neighbors
```

먼저 import를 통해 모듈을 가져옵니다. knn함수를 만들고 트레이닝, 테스트, k 값을 입력인수로 받습니다. 인스턴스간 거리는 유클리디언 거리를 사용합니다.

```
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)

    return math.sqrt(distance)
```

유클리디언 거리함수를 만들고 인스턴스와 거리를 입력인수로 받습니다. pow함수는 제곱수를, sqrt는 루트를 계산해줍니다.

```
def centroid(neighborcoor,k,length):
    centro=[]   
    for x in range(length):
        mean_coor=0
        for y in range(k):
            mean_coor += neighborcoor[y][x]
        centro.append(mean_coor/k)
    return centro
```

인스턴들의 중심을 계산합니다.(mean vector를 사용하기 위함)

```
def meanDistance(neighborcoor,testInstance,k):
    meandist = []
    length = len(testInstance)-1
    center = centroid(neighborcoor,k,length)
    print('center:',center)
    dist = euclideanDistance(testInstance,center,length)
    meandist.append(dist)
    return meandist
```

평균거리를 계산하는 함수를 만들고 트레이닝 데이터에서의 이웃, 테스트 데이터, k값을 입력인수로 받습니다.

```
def main():

    trainingSet = [[1,2,'-'],[2,2,'-'],[3,2,'-'],[4,2,'-'],[1,1,'-'],[2,1,'-'],[3,1,'-'],[4,1,'-']]
    testSet = [[2.5,2.5,'-'],[0.7,1,'-']]
    
    k = 4

    neighbors_inform=[]
    knn_dist_mean=[]
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        neighbors=np.array(neighbors)
        neighbors_inform.append(neighbors)
        
    print('neighbors_inform:',neighbors_inform)
    
    neighbors_dist=np.array(neighbors_inform)[:,:,1]
    knn_dist_max=np.max(neighbors_dist,axis=1)
    knn_dist_avg=np.mean(neighbors_dist,axis=1)
    neighborcoordinate = np.array(neighbors)[:,0]
    meanDist = meanDistance(neighborcoordinate,testSet[x],k)
    knn_dist_mean.append(meanDist)
    
    print('knn_dist_max:',knn_dist_max)
    print('knn_dist_avg:',knn_dist_avg)
    print('knn_dist_mean:',knn_dist_mean) 

    
main()
```

```
출력결과
neighbors_inform: [array([[list([2, 2, '-']), 0.7071067811865476],
       [list([3, 2, '-']), 0.7071067811865476],
       [list([1, 2, '-']), 1.5811388300841898],
       [list([4, 2, '-']), 1.5811388300841898]], dtype=object), array([[list([1, 1, '-']), 0.30000000000000004],
       [list([1, 2, '-']), 1.044030650891055],
       [list([2, 1, '-']), 1.3],
       [list([2, 2, '-']), 1.6401219466856727]], dtype=object)]
center: [1.5, 1.5]
knn_dist_max: [1.5811388300841898 1.6401219466856727]
knn_dist_avg: [1.1441228056353687 1.071038149394182]
knn_dist_mean: [[0.9433981132056605]]
```

트레이닝과 테스트 데이터, k값은 4로 입력합니다.
테스트 데이터에서 첫번째 데이터는 그림에서 "빨간색원"에 해당하며, 두번째 데이터는 그림에서 "빨간색삼각형"에 해당합니다.
거리는 max와 avg, mean 방법을 사용하여 계산하였습니다.
출력된 순서는 다음과 같습니다.
1.knn_dist_max는 가장 먼거리를 novelty score로 표현
2.knn_dist_avg는 전체 거리의 평균을 novelty score로 표현
3.knn_dist_mean는 점들의 mean vector와 인스턴스와의 거리를 Novelty distance로 표현

```
import math
import numpy as np
import operator
from collections import namedtuple 
import matplotlib.pyplot as plt 
import random


Point = namedtuple('Point', 'x y')

class ConvexHull(object):  
    _points = []
    _hull_points = []

    def __init__(self):
        pass

    def add(self, point):
        self._points.append(point)

    def _get_orientation(self, origin, p1, p2):
        '''
        Returns the orientation of the Point p1 with regards to Point p2 using origin.
        Negative if p1 is clockwise of p2.
        :param p1:
        :param p2:
        :return: integer
        '''
        difference = (
            ((p2.x - origin.x) * (p1.y - origin.y))
            - ((p1.x - origin.x) * (p2.y - origin.y))
        )

        return difference
```

import를 통해 모듈을 가져옵니다. Novelty Dectection을 하기위한 Convec Hull과의 거리 측정을 위해 먼저 Convex Hull의 좌표를 구합니다. 방향을 정하기 위한 함수를 만들고 포인트들을 입력인수로 받습니다. difference가 양수이면 시계 방향, 음수이면 반시계 방향으로 정하게됩니다.

```
    def compute_hull(self):
        '''
        Computes the points that make up the convex hull.
        :return:
        '''
        points = self._points

        # get leftmost point
        start = points[0]
        min_x = start.x
        for p in points[1:]:
            if p.x < min_x:
                min_x = p.x
                start = p

        point = start
        self._hull_points.append(start)

        far_point = None
        while far_point is not start:
            
            # get the first point (initial max) to use to compare with others
            p1 = None
            for p in points:
                if p is point:
                    continue
                else:
                    p1 = p
                    break

            far_point = p1
 
            for p2 in points:
                # ensure we aren't comparing to self or pivot point
                if p2 is point or p2 is p1:
                    continue
                else:
                    direction = self._get_orientation(point, far_point, p2)
                    if direction > 0:
                        far_point = p2


            self._hull_points.append(far_point)
            point = far_point
```

가장 왼쪽에 있는 데이터를 시작 포인트로 지정합니다. 그런다음 시작지점(Origin)을 제외한 far_point인 p1, Origin과 far_point를 제외한 p2를 random하게 정합니다. 계산하여 반시계방향(음수값)이면, p2를 final point로 복사 즉, 새로운 far_point로 정합니다.

```
    def get_hull_points(self):
        if self._points and not self._hull_points:
            self.compute_hull()

        return self._hull_points
```
```
    def display(self):
        # all points
        x = [p.x for p in self._points]
        y = [p.y for p in self._points]
        plt.plot(x, y, marker='D', linestyle='None')

        # hull points
        hx = [p.x for p in self._hull_points]
        hy = [p.y for p in self._hull_points]
        plt.plot(hx, hy)

        plt.title('Convex Hull')
        plt.show()
```

위는 Convex Hull을 시각적으로 보여주기 위한 코드입니다.

```
def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)
```

데이터 점과 선과의 거리를 측정하기 위한 함수를 정의합니다. 데이터 점과 선과의 거리를 구하고 t<0이면 t=0으로, t>1이면 t=1로 정합니다.

```
def getconvexdist(arrtestSet,getConvexHullPoint):

    pnt2lineset=[]
    for i in range(arrtestSet.shape[0]):
        for j in range(getConvexHullPoint.shape[0]-1):
        #pnt2line(pnt, getConvexHullPoint[i], getConvexHullPoint[i+1])
            result=pnt2line(arrtestSet[i], getConvexHullPoint[j], getConvexHullPoint[j+1])
            pnt2lineset.append(result[0])

    pnt2lineset=np.reshape(pnt2lineset,(arrtestSet.shape[0],getConvexHullPoint.shape[0]-1))
    convex_dist=np.min(pnt2lineset,axis=1)
    print('pnt2lineset:',pnt2lineset)

    return convex_dist
```

데이터와 Convex Hull까지의 거리를 측정하기 위한 함수를 정의합니다.

```
def dot(v,w):
    x,y = v
    X,Y = w
    return x*X + y*Y

def length(v):
    x,y = v
    return math.sqrt(x*x + y*y)

def vector(b,e):
    x,y = b
    X,Y = e
    return (X-x, Y-y)

def unit(v):
    x,y = v
    mag = length(v)
    return (x/mag, y/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y = v
    return (x * sc, y * sc)

def add(v,w):
    x,y = v
    X,Y = w
    return (x+X, y+Y)
```
```
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
```

앞의 코드들과 동일하게 유클리디어 거리 함수를 정의합니다.

```
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    return neighbors
```

knn함수를 만들고 트레이닝, 테스트, k 값을 입력인수로 받습니다. 인스턴스간 거리는 유클리디언 거리를 사용합니다.

```
def main():  
    ch = ConvexHull()


    trainingSet = [[1,2,'-'],[2,2,'-'],[3,2,'-'],[4,2,'-'],[1,1,'-'],[2,1,'-'],[3,1,'-'],[4,1,'-']]
    testSet = [[2.5,2.5,'-'],[0.7,1,'-']]

    arrtrainingSet=np.array(trainingSet)
    arrtestSet=np.array(testSet)

    arrtrainingSet=arrtrainingSet[:,:-1].astype(np.float)
    arrtestSet=arrtestSet[:,:-1].astype(np.float)
  
    for i in range(arrtrainingSet.shape[0]):
        ch.add(Point(arrtrainingSet[i,0], arrtrainingSet[i,1]))


    getConvexHullPoint=np.array(ch.get_hull_points())
    ch.display()

    print('ConvexHullPoint:',getConvexHullPoint)

    convex_dist=getconvexdist(arrtestSet,getConvexHullPoint)

    print('convex_dist:',convex_dist)
        
    k = 4
    neighbors_inform=[]
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        neighbors_inform.append(neighbors)
        
    neighbors_dist=np.array(neighbors_inform)[:,:,1]
    knn_dist_avg=np.mean(neighbors_dist,axis=1)
    
    #print('neighbors_dist:',neighbors_dist)
    print('knn_dist_avg:',knn_dist_avg)
    
    knn_dist_hybrid=knn_dist_avg*2/(1+np.exp(-convex_dist))
    print('knn_dist_hybrid:',knn_dist_hybrid)
    
    main()
```
```
출력결과
ConvexHullPoint: [[1. 2.]
 [1. 1.]
 [2. 1.]
 [3. 1.]
 [4. 1.]
 [4. 2.]
 [1. 2.]]
pnt2lineset: [[1.58113883 1.58113883 1.5        1.58113883 1.58113883 0.5       ]
 [0.3        0.3        1.3        2.3        3.3        1.04403065]]
convex_dist: [0.5 0.3]
knn_dist_avg: [1.1441228056353687 1.071038149394182]
knn_dist_hybrid: [1.4243398328171621 1.2304997002785911]
```

![그림8.1](https://i.imgur.com/EBUtyH4.png)

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

아래는 K-Means clustering-Novelty Detection의 파이썬 코드를 나타낸 것입니다.

```
import numpy as np
import random
import csv
import matplotlib.pyplot as plt


def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(2):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
```

모듈을 import하고 csv파일로 데이터를 부르는 함수를 정의합니다.

```
def main():
    
    trainingSet=[]
    testSet=[]
    split = 1
    random.seed(100)
    loadDataset('C:/Users/IIS_LAB/Documents/수업/2018-2/비즈니스어낼리틱스/과제/ND_KM.csv', split, trainingSet, testSet)
    trainX=np.array(trainingSet)
    
    print('Train set: ' ,repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    print('Train set: ' ,trainingSet[0:10])
    a=trainX[:,:-1].astype(np.float)
    plt.scatter(a[:,0], a[:,1]);

main()
```

```
출력결과
Train set:  1509
Test set: 0
Train set:  [[1.996251169, 0.931077233, '1'], [-1.298157109, -2.322357985, '1'], [1.844962177, -1.774005487, '1'], [-0.266268692, -1.735327618, '1'], [-2.265353559, -0.367262574, '1'], [-0.089113845, -1.810973134, '1'], [-1.603322368, -1.983500956, '1'], [-2.054007548, -1.846174646, '1'], [1.726899216, 1.417497251, '1'], [1.786822248, 2.201596215, '1']]
```
![그림11.1](https://i.imgur.com/4IK8bN3.png)

csv 형식인 데이터 파일을 로드하고 plot을 그려 확인합니다.

```
def kmeans(X,k,maxIt):
    numPoints,numDim=X.shape

    dataSet=np.zeros((numPoints,numDim+1))
    dataSet[:,:-1]=X
    centroids=dataSet[np.random.randint(numPoints,size=k),:]
    centroids[:,-1]=range(1,k+1)
```

K-Means clustering을 위한 함수를 정의합니다. 첫번째 중심(centroid)를 랜덤하게 배정합니다.

```
def main():

    trainingSet=[]
    testSet=[]
    split = 1
    random.seed(100)
    loadDataset('C:/Users/IIS_LAB/Documents/수업/2018-2/비즈니스어낼리틱스/과제/ND_KM.csv', split, trainingSet, testSet)
    trainX=np.array(trainingSet)

    k=3
    max_iter=100
    final_result=kmeans(trainX[:,:-1],k,max_iter)

    main()
```

위에서 확인한 데이터를 csv파일로 로드하여 트레이닝과 테스트 데이터로 랜덤하게 분할합니다.
k의 값은 3(군집수 3개)으로 주며, 중심점 조정횟수를 100회로 제한합니다.

```
def kmeans(X,k,maxIt):
    numPoints,numDim=X.shape

    dataSet=np.zeros((numPoints,numDim+1))
    dataSet[:,:-1]=X
    centroids=dataSet[np.random.randint(numPoints,size=k),:]
    centroids[:,-1]=range(1,k+1)

    iterations=0;
    oldCentroids=None
    
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        oldCentroids=np.copy(centroids)
        iterations+=1
        updateLabels(dataSet, centroids)
        centroids=getCentroids(dataSet, k)
    return dataSet, centroids

def shouldStop(oldCentroids,centroids,iterations,maxIt):
    if iterations>maxIt:
        return True
    return np.array_equal(oldCentroids, centroids)
```

3개의 중심점을 기준으로 각 데이터들이 군집에 할당되도록 합니다.
또한, 새로이 업데이트되는 중심점이 기존의 중심점과 차이가 없을때까지 반복 작업을 수행합니다.

```
def updateLabels(dataSet,centroids):
    numPoints,numDim=dataSet.shape
    for i in range(numPoints):
        dataSet[i,-1]=getLabelFromClosestCentroid(dataSet[i,:-1], centroids)

def getLabelFromClosestCentroid(dataSetRow,centroids):
    label=centroids[0,-1]
    minDist=np.linalg.norm(dataSetRow-centroids[0,:-1])
    for i in range(1,centroids.shape[0]):
        dist=np.linalg.norm(dataSetRow-centroids[i,:-1])
        if dist<minDist:
            minDist=dist
            label=centroids[i,-1]
    return label
```

군집에 할당된 데이터들의 label(어느 군집에 할당되었는지)를 정하는 함수를 정의합니다.

```
def getCentroids(dataSet,k):       
    result=np.zeros((k,dataSet.shape[1]))

    for i in range(1,k+1):
        oneCluster=dataSet[dataSet[:,-1]==i,:-1]
        result[i-1,:-1]=np.mean(oneCluster,axis=0)
        result[i-1,-1]=i
    return result
```

새로운 중심점을 구하기위한 함수를 정의합니다.
하나의 군집에서 데이터들 간의 평균을 구하여 중심점을 새로이 할당합니다.

```
def shouldStop(oldCentroids,centroids,iterations,maxIt):
    if iterations>maxIt:
        return True
    return np.array_equal(oldCentroids, centroids)
```

각 군집의 중심점의 변화가 없을때까지 반복하는 함수를 정의합니다.
이 함수에서 반복의 변화가 없을시에 상기에서 정의한 코드에 따라 중심점의 반복 할당 작업이 중지됩니다.

```
def getNoveltyScore(dataSetRow,centroids,k):
    minDist=[]    
    for i in range(k):
        Dist=np.linalg.norm(dataSetRow[:,:-1]-centroids[i,:-1],axis=1)   
        minDist.append(Dist)        
    NoveltyScore=np.min(minDist,axis=0)
    return NoveltyScore
```

Novelty Score를 계산하는 함수를 정의합니다.
데이터들의 최근접 군집의 중심점과의 거리를 이용하여 Novelty Score를 계산합니다.

```
def main():
    
    trainingSet=[]
    testSet=[]
    split = 1
    random.seed(100)
    loadDataset('C:/Users/IIS_LAB/Documents/수업/2018-2/비즈니스어낼리틱스/과제/ND_KM.csv', split, trainingSet, testSet)
    trainX=np.array(trainingSet)
    
    print('Train set: ' ,repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    print('Train set: ' ,trainingSet[0:10])
    a=trainX[:,:-1].astype(np.float) 
    k=3
    max_iter=100
    final_result=kmeans(trainX[:,:-1],k,max_iter)
    
    kmeans_result=final_result[0]
    centroid_result=final_result[1]
    Score=getNoveltyScore(kmeans_result,centroid_result,k)
    
    print('final cluster:',kmeans_result)
    print('final centroid:',centroid_result)
    print('Novelty Score:',Score)
    
    x=kmeans_result[:,0]
    y=kmeans_result[:,1]
    
main()
```
```
출력결과
Train set:  1509
Test set: 0
Train set:  [[1.996251169, 0.931077233, '1'], [-1.298157109, -2.322357985, '1'], [1.844962177, -1.774005487, '1'], [-0.266268692, -1.735327618, '1'], [-2.265353559, -0.367262574, '1'], [-0.089113845, -1.810973134, '1'], [-1.603322368, -1.983500956, '1'], [-2.054007548, -1.846174646, '1'], [1.726899216, 1.417497251, '1'], [1.786822248, 2.201596215, '1']]
final cluster: [[ 1.99625117  0.93107723  2.        ]
 [-1.29815711 -2.32235799  2.        ]
 [ 1.84496218 -1.77400549  2.        ]
 ...
 [ 3.          0.          2.        ]
 [ 4.         -1.8         2.        ]
 [ 5.7         2.          3.        ]]
final centroid: [[-6.49875619  3.46631533  1.        ]
 [ 0.04940347 -0.09926152  2.        ]
 [ 4.34440535  4.42479706  3.        ]]
Novelty Score: [2.20268334 2.59963025 2.45536116 ... 2.95226569 4.30113058 2.77799885]
```

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

아래는 PCA-Novelty Detection의 파이썬 코드를 나타낸 것입니다.

```
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import random
import csv


def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(2):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
```

모듈을 import합니다. (PCA는 sklearn 패키지의 함수를 사용합니다.)
데이터는 csv 형식의 파일을 로드해서 사용하며, 트레이닝과 테스트 데이터로 분할합니다.

```
def main():

	# prepare data
    trainingSet=[]
    testSet=[]
    split = 0.8
    random.seed(100)
    loadDataset('C:/Users/IIS_LAB/Documents/수업/2018-2/비즈니스어낼리틱스/과제/ND_PCA.csv', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    

    trainX=np.array(trainingSet)
    X=trainX[:,:-1].astype(np.float)

    pca = PCA(n_components=1)
    pca.fit(X)
    X_pca = pca.transform(X)
    print("original shape:   ", X.shape)
    print("transformed shape:", X_pca.shape)

    X_new = pca.inverse_transform(X_pca)
    Score = np.linalg.norm(X-X_new,axis=1)
    
    
    plt.scatter(X[:, 0], X[:, 1], alpha=0.8,s=Score*100)
    plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
    plt.axis('equal');
    
    
main()
```

트레이닝과 훈련데이터를 8:2 비율로 나눕니다.
PCA 함수를 사용하여 1차원으로 축소합니다.
Original shape는 데이터의 원래 행열을 나타내고, transformed shape는 1차원으로 축소한 것을 나타냅니다.

```
출력결과
Train set: 167
Test set: 38
original shape:    (167, 2)
transformed shape: (167, 1)
```
![그림15.1](https://i.imgur.com/qx0d7jl.png)
