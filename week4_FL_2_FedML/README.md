# Week 4. FedML & FedOps  
---  

## 1. FedML: 연합학습 프레임워크  

- flower와 차이점: fedml은 format을 맞춰줘야함. 그런 의미에서 flower는 비교적 형식에 자유로움.


- 예제: fedavg 관련 로지스틱 회귀 예제 (경로: FedML\python\examples\simulation\sp_fedavg_mnist_lr_example)  

<br />

## 2. FedOps: 연합학습 수명주기 관리 및 운영

<img src="https://user-images.githubusercontent.com/11987128/236702243-9152e51d-726e-443b-93bd-01986a047dd4.png">  
출처: 강의자료 (https://github.com/Kwangkee/Gachon/blob/main/slides/FLScalize_%EC%96%91%EC%84%B8%EB%AA%A8_20230331.pdf)

<br/>

- FedOps: 기존 MLOps를 연합학습에 맞게 확장시킨 개념  
- manager 컴포넌트: 연합학습 모니터링 및 server, client 관리  
- Client Manager가 비동기 방식(Asynchronous)인 이유는 유사시에도 지속적으로 작동하기 위해서임 (예: 노드 중 하나가 갑자기 꺼지거나 서버가 다운된 경우에 동기 방식으로 하면 문제 발생)

- FLScalize
     - 지속적으로 client/server를 확인하고 관리하며 통합/배포가 가능함
- elastic search: 클라이언트 모니터링 대시보드

<br />

## 3. FLScalize 실습

- 도커에서 client 로그 확인 및 server manager 실행까지 되었으나 elastic search에 결과가 보이지 않음 (로그가 제대로 쌓이지 않는 오류인듯함)  

<img src="https://user-images.githubusercontent.com/11987128/236702027-e578f419-61a2-4b93-a127-39fb377db20e.png">

<img src="https://user-images.githubusercontent.com/11987128/236702040-a0e100a8-5f61-4ac7-934f-e070d40f7ca5.png">

<img src="https://user-images.githubusercontent.com/11987128/236702120-aebe1542-4960-4bcd-a1b8-6522d491ab61.png">  