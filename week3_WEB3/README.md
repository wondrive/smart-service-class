# 3주차: WEB 3.0과 이더리움
<br/>

## 수업내용
- 이론
  - 탈중앙화를 요점으로 하는 새로운 웹 환경 'WEB 3.0'의 도래와 그 대표격 사례인 암호화폐, 그 중에서 특히 이더리움의 생태계를 학습함.  
  - 투자자산으로만 여기던 블록체인 기반 암호화폐의 탄생 배경과 거래 방식을 살펴보니 흥미로웠음.  
  - 수업에서 새로이 알게 된 내용이 많아 한 눈에 보기 위해 `note.md`에 정리하였음.

- 실습
  - 지갑을 생성하고, 나만의 토큰을 발행해 배포하였음

<br/>
<br/>

## 수업에서 생긴 궁금증 및 해결
Q1. Smart Contract가 없는 비트코인은 어떤 한계가 있었는지?  
A. 비트코인도 스마트 컨트랙트가 가능하나 구현에 제약이 많음. 이외에도 튜링 불완전성, 어려운 확장성 등이 이더리움과의 큰 차이임  
(출처: [업비트 투자자보호센터](https://m.upbitcare.com/academy/education/coin/253))  
<br/>

Q2. 메타버스가 왜 WEB 3.0 환경에 속하는지와, 메타버스가 블록체인과 어떠한 공통점이 있는지 궁금함. 메타버스와 기존 게임의 차이를 모르겠음.  
A. 메타버스는 블록체인 기반의 가산자산(NFT, 암호화폐 등)을 포괄하며, 이러한 가상경제체제는 WEB 3.0의 특징인 읽기, 쓰기, 소유 중 '소유'를 만족시킨다.  
<br/>

Q3. 이더리움 특징 중 '세계 컴퓨터'의 의미가 이해가지 않음. 모든 컴퓨터가 서버가 된다는 의미인지?  
A. 모든 노드(컴퓨터)들이 OS에 관계없이 EVM에서 똑같은 데이터로 똑같은 연산을 수행하므로 동일한 상태가 된다.  
즉, 모든 노드가 동일한 하나의 컴퓨터를 사용하는 것과 같다. '세계 컴퓨터'의 장점은 하나의 노드가 해킹당해도 결국 다른 노드가 작업을 기억하므로 공격에 강하다는 점이다.  
<br/>

Q4. NFT와 비트코인의 차이?  
A. 코인과 토큰의 차이와 같다. 비트코인은 코인에 해당하고, NFT는 토큰에 해당하며 코인처럼 현금화가 가능하지 않다. 또한 NFT는 독자적 블록체인이 없다는 점도 큰 차이다.  

<br/>
<br/>

## 실습: 나만의 `Won` 토큰 발행
**1. Meta mask, Kaikas 지갑 생성**

|Meta mask 지갑|Kaikas 지갑|
|:---:|:---:|
|<img width="50%" src="https://user-images.githubusercontent.com/11987128/236323791-0349c1b6-115d-4a34-b76c-58705a53e76f.JPG"/>|<img width="70%" src="https://user-images.githubusercontent.com/11987128/236323792-b369bd1c-6982-4414-9bdc-2a6cb18716c5.JPG"/>|

<br/>

**2. Remix에서 Storage 컨트랙트 배포하기**  

i. 테스트 네트워크에 코인 수급
- 테스트 이더리움 수급 (Goerli 네트워크, 수급사이트: GOERLI FAUCET, Ethereum Goerli Faucet)  
<img width="60%" src="https://user-images.githubusercontent.com/11987128/236323781-20ca9d1a-145f-4bc3-937a-8bd9397b8104.JPG"/>
아쉽게도 메인 네트워크에 있어야 할 최소한의 이더리움 잔액 조건을 충족하지 못해 테스트 이더는 수급받지 못했음.  

- 테스트 클레이튼 수급 (Baobab 네트워크, 수급사이트: Klaytn Wallet)  
<img width="60%" src="https://user-images.githubusercontent.com/11987128/236323787-5604d1a5-30e7-43ea-acd1-7c3a7551e213.JPG"/>

<br/>

ii) Remix에서 Storage 컨트랙트 배포하기

- 컨트랙트 검색하기
[테스트 Baobab 사이트](https://baobab.scope.klaytn.com/)에서 Klaytn 컨트랙트 검색 (hash로 검색 가능)

- 'Won'이라는 이름의 토큰 발행
<img width="100%" src="https://user-images.githubusercontent.com/11987128/236323794-94379bd8-20ad-4142-8cd8-1fc9114100a0.JPG"/>
<img width="40%" src="https://user-images.githubusercontent.com/11987128/236323796-607e9222-6e95-47e1-8ee4-28a13872c768.JPG"/>

<br/>

iii) 토큰 전송

- 컨트랙트 주소를 통해 토큰 전송
<img width="40%" src="https://user-images.githubusercontent.com/11987128/236323798-5de64a96-48d9-490c-9bd8-fdc464667ab5.JPG"/>
