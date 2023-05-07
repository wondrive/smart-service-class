# Week 2. 연합학습 (Federated Learing)

---

### 이론   

<br/>

**1. 개념**
<img width="60%" style="margin-left:auto; margin-right:auto; display:block;" src="https://upload.wikimedia.org/wikipedia/commons/1/11/Centralized_federated_learning_protocol.png">  

- 다수의 기기(클라이언트)의 로컬 모델에서 학습된 내용을 서버로 모아 글로벌 모델을 만드는 것  
- 장점: 분산학습이 가능하면서 데이터 보호를 할 수 있음.  
- 활용 분야: 데이터 보안이 중요한 의료 영역에서 가장 활발하게 적용되고 있으며, 자동차 주행 정보, 스마트폰 앱 사용 등 다양한 영역에 걸쳐 활용되고 있음.  
- FL 오픈소스 프레임워크: Flower, FedML, FedScale, Pysyft, TFF 등..
     - Flower
          - 특징: 개발 언어 및 머신러닝 프레임워크에 구애 받지 않음
          - 장점: 쉬운 사용성 및 확장성 
     - FedScale
          - 특징: 벤치마크에 집중, 포괄적이고 현실적인 데이터셋 제공
          - 자동화된 평가 플랫폼 제공 (FedScale Runtime)
          
<br/>

**2. FL 작동 순서**

> step 1. Server는 학습 모델 선정 (e.g. ResNet, MobileNet, ..)  
> step 2. Server는 Client 모델에 초기 weight 배포  
> step 3. Client는 로컬 데이터를 가지고 로컬 모델 학습  
> step 4. 각 Clients는 로컬 모델의 weight를 서버로 전송  
> step 5. Server는 글로벌 모델 업데이트(=aggregation)  

<img width="80%" style="margin-left:auto; margin-right:auto; display:block;" src="https://2603032841-files.gitbook.io/~/files/v0/b/gitbook-legacy-files/o/assets%2F-LkwjM9iFxyVX0PINZ87%2F-LrXYKQk6zb8DtDHQDWr%2F-LrXjxgPaO7YA_fdB4Uf%2FScreen%20Shot%202019-10-19%20at%202.41.08%20AM.png?alt=media&token=bc5fd8a8-7db2-416e-8203-c1833cf5439c">
출처: Wei Yang Bryan Lim et al., Federated Learning in Mobile Edge Networks: A Comprehensive Survey, 2019, https://doi.org/10.48550/arXiv.1909.11875

<br />

**3. 주요 알고리즘**
- FedAvg: 로컬 모델들의 가중치 평균을 구해 글로벌 모델의 가중치 설정  
- FedProx
     - 규제를 위한 Proxiaml term를 클라이언트에 추가
     - 로컬 데이터의 편향이 심한 클라이언트는 라운드를 제한적으로 참여 (예: round=10으로 정해도 전부 돌지 않음.)
     - FedAvg와 차이점: 모든 클라이언트를 동등하게 고려하지 않음

<br />

**4. FL 연구분야**
- Client Selection/Incentive Mechanism: 전체 클라이언트 중 좋은 클라이언트만 활용하겠다  
- Model Aggregation optimization: 글로벌 모델에 weight를 합칠 때 어떤 weight를 합칠지 (대표적 예: FedProx)
- Personalization: 로컬 데이터를 어떻게 학습시킬지 연구하는 분야

<br/>
<br/>

---

### 실습
Flower framework를 사용한 FedProx 실습
<br/>

**1. 실습 코드**
     - git clone --depth=1 https://github.com/adap/flower.git
     - fed avg 실습 경로: `flower/baselines/flwr_baselines/publications/fedavg_mnist`
     - fed prox 실습 경로: `flower/baselines/flwr_baselines/publications/fedprox_mnist`  

<br/>

**2. fed prox 실습 이미지**

- 주요 하이퍼파라미터: straggler(γ), mu(μ)
<img src="https://user-images.githubusercontent.com/11987128/236694480-60d087ca-64ef-4bef-8f89-1d0dacdfae9c.png">
<img src="https://user-images.githubusercontent.com/11987128/236694500-9a1a511c-8ad5-4aa8-999e-8e1b6c312101.png">

- 결과
<img src="https://user-images.githubusercontent.com/11987128/236694512-c124b0d2-7b83-4fe1-b787-d90137bf9349.png">

