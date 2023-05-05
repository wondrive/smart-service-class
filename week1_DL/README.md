# Homework

- 숙제 내용
     - Common.py 내의 블록/모델을 활용하여 우수한 성능의 network 만들어 보기  
     - 사용한 블록/모델의 관련 논문 서치 및 장/단점 기록  
     - Integrated_gradients를 이용하여 exp3의 최종 산출물과 비교해보기  
<br/>

---  
- VGG, ResNet등 CNN 기반 역대 SOTA 모델에서 MNIST 학습시 수렴이 늦고 정확도가 떨어지는 경향이 있었음. MNIST 이미지가 단순하고 사이즈가 작아서(28x28) 그런 것으로 생각됨.
- 예: 다음은 VGG16에서 MNIST를 학습한 결과임  
     <img width="30%" src="https://user-images.githubusercontent.com/11987128/236466456-c0bc31a9-c88c-4cfd-984d-abd39a274002.PNG"/>  
- 따라서 5개 이하의 conv층을 가진 단순한 모델을 `common.py`에 구현한 것으로 실습하였음
<br/>


### 1. VGG 블록 활용  
- 참고 모델: VGG16  

     ||내용|  
     |:----:|---|  
     |구조|<img width="70%" src="https://iq.opengenus.org/content/images/2019/01/vgg_layers.png"/><br/>출처: https://iq.opengenus.org/vgg16/|
     |특징|커널 사이즈를 3x3으로 고정|
     |장점|-작은 커널을 여러 번 사용하여 큰 필터(5x5, 7x7)를 한 번 사용하는 것보다 작은 파라미터를 가지고 비슷한 성능을 냄<br/>- 레이어를 더 깊게 쌓을 수 있음<br/>- 더 많은 비선형 함수(활성화 함수) 적용|
     |단점|FC Layer가 3개라서 파라미터가 너무 많음|  
     
- hyperparamaeter
     - epoch: 10
     - input_size: 32x32
     - batch_size: 200

- 실험 결과
     -  Run summary:  
          wandb:       acc 0.99297  
          wandb:      loss 1.46958  
          wandb:  test_acc 0.9912  
          wandb: test_loss 1.47173  
<br/>



### 2. BottleNeck 블록 활용   
- BottleNeck을 활용한 대표적 모델 ResNet  
     ||내용|  
     |:---:|---|  
     |구조|<img width="80%" src="https://media.licdn.com/dms/image/C5112AQHm-WWP3Tb-yw/article-cover_image-shrink_600_2000/0/1583950796769?e=2147483647&v=beta&t=Jl6tX_PaEwWrQaOC7kKXzFgmNH2pR7wj-XNN1TWD5RQ"><br/>출처: Kaiming He, et al., Deep Residual Learning for Image Recognition, CVPR, 2016|
     |특징|잔차 연결: 앞 블록의 output을 뒤의 블록에 element-wise add 해줌|
     |장점|망이 깊어져도 정보 손실이 적으며 기울기 소실 문제를 완화함|
     |단점|망이 깊어질수록 연산량 대폭 증가 (레이어 약 1000개 기준)| 

- hyperparamaeter
     - epoch: 10
     - input size: 32*32
     - batch_size: 200

- 실험 결과
     - Run summary:  
     wandb:       acc 0.99203  
     wandb:      loss 1.46998  
     wandb:  test_acc 0.9874  
     wandb: test_loss 1.47435  
<br/>



### 3. Transformer 블록 활용  

- Transformer
     ||내용|  
     |:---:|---|  
     |구조|<img width="50%" src="https://production-media.paperswithcode.com/method_collections/VIT.png"/><br/>출처: Salman Khan et al, Transformers in Vision: A Survey, 2021, https://doi.org/10.1145/350524|
     |특징|Transformer의 인코더 부분을 사용하며, Self-Attention 매커니즘을 적용함|
     |장점|- 모델 사이즈를 키울수록 성능이 보장됨<br/>- 전이학습시 CNN보다 연산량 적음|
     |단점|- 중간 사이즈의 데이터셋으로 학습시 ResNet보다 낮은 성능을 보임<br/>- inductive bias가 부족하여 일반화 능력, 즉 처음 보는 입력에 대한 예측 능력이 떨어지므로 CNN보다 더 많은 데이터 필요함| 

- hyperparamaeter
     - epoch: 10
     - input_size: 32x32
     - batch_size: 50 (메모리 부족으로..)

- 실험 결과
     - Run summary:  
     wandb:        acc 0.8947  
     wandb:      loss 1.56638  
     wandb:  test_acc 0.8946  
     wandb: test_loss 1.56677  
     
- 아쉬운 점
     - 층이 깊을수록 장점이 극대화 되는 모델을 너무 얕은 층으로 테스트 한 것이 아쉬움
     - 한편으로는 훈련 속도가 느리고 연산량이 CNN보다 많아서 간단한 실습 활동에 활용하기에는 덜 매력적이었음
     - 현실세계의 이미지 데이터셋은 MNIST보다 복잡하고 사이즈도 크므로 어떨지 궁금함
<br/>



### 4. CSP BottleNeck 블록 활용  
- CSP BottleNeck
     ||내용|  
     |:---:|---|  
     |구조|<img width="70%" src="https://user-images.githubusercontent.com/47938053/90522132-eff2c000-e19d-11ea-94d9-964a81e92280.jpg"/><br/>출처: Chien-Yao Wang et al., CSPNet: A New Backbone that can Enhance Learning Capability of CNN. 2019, https://doi.org/10.48550/arXiv.1911.11929|
     |특징|이전 레이어의 정보를 뒤로 전달해준다는 점에서 ResNet과 비슷하지만 concat 연산이라는 점에서 차이가 있음|
     |장점|- 정확도를 유지하면서 모델을 경량화 할 수 있음<br/>- 메모리 비용 감소<br/>- 연산에서의 병목을 줄일 수 있음|
     |단점|x| 

- hyperparamaeter
     - epoch: 10
     - input_size: 32x32
     - batch_size: 100

- 실험 결과
     - Run summary:  
     wandb:       acc 0.99125
     wandb:      loss 1.47022
     wandb:  test_acc 0.987
     wandb: test_loss 1.47468
<br/>



### 5. SPPF 블록 활용  

- SPP Net
     ||내용|  
     |:---:|---|  
     |구조|<img width="70%" src="https://production-media.paperswithcode.com/methods/new_teaser_dU3j8iq.jpg"/><br/>출처: Kaiming He, et al., Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition, 2015, https://doi.org/10.48550/arXiv.1406.4729|
     |특징|- 입력 이미지의 크기에 관계없이 Conv layer를 통과시키고 FC layer 전에 동일한 크기로 조절해주는 Pooling 적용함<br/>- 이미지를 여러 영역으로 나눠 각 영역별 특성을 파악함|
     |장점|- 네트워크 속도를 향상하고, 피처맵 크기를 줄이면서도 receptive field는 크게 유지하여 정보를 보존함<br/>- CNN에 다양한 크기의 이미지 입력하여 고정된 크기의 피처 출력 가능|
     |단점|x| 

- hyperparamaeter
     - epoch: 10
     - input_size: 32x32
     - batch_size: 100

- 실험 결과
     - Run summary:
     wandb:       acc 0.97575
     wandb:      loss 1.4861
     wandb:  test_acc 0.9602
     wandb: test_loss 1.50099
---

<br/><br/>

# 결론  

**1. 결과 요약**

<img width="100%" src="https://user-images.githubusercontent.com/11987128/236466461-9997013a-d6c3-4a1e-be8c-dbcf23bc0bb8.PNG"/>

| |VGG Block|BottleNeck|TransformerBlock|CSPBottleNeck|SPPF|  
|:---:|---|---|---|---|---|    
|**acc**|0.99297|0.99203|0.8947|0.99125|0.97575|
|**loss**|1.46958|1.46998|1.56638|1.47022|1.4861|
|**test_acc**|0.9912|0.9874|0.8946|0.987|0.9602|
|**test_loss**|1.47173|1.47435|1.56677|1.47468|1.50099|

- 성능: VGG Block을 모방한 단순한 구조가 가장 정확도가 높고 에러가 낮았음.
- MNIST 데이터셋이 크기가 작고 이진 데이터로 단순하기 때문에 단순한 모델이 가장 효율적이라고 생각됨.
- GPU 및 메모리 자원 활용: TransformerBlock가 메모리 점유시간과 할당량이 가장 많았으며, GPU의 활용도 가장 높았음

<br/>

**2. heatmap 비교**  
Integrated_gradients를 이용하여 성능이 가장 좋았던 VGG Block과 exp3의 최종 산출물 비교  

|exp 3|VGG Block|  
|---|---|  
|<img width="100%" src="https://user-images.githubusercontent.com/11987128/236466465-9a40497b-091e-4517-bddd-cfd5e2c2e353.PNG"/>|<img width="100%" src="https://user-images.githubusercontent.com/11987128/236466471-c16c4cfa-d9ae-493c-8e40-95dcbd12cc4f.png"/>|  
