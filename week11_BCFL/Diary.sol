/*
실습 메모: Remix(https://remix.ethereum.org/) 접속하여 다음 수행

1) contract 폴더에 이 소스파일(Diary.sol) 생성 (파일명도 똑같이 맞춰주기)
   Ctrl_s 저장하면 알아서 컴파일 됨

2) 네 번째 deploy 메뉴 클릭 > Deploy 버튼 누르기

3) Deployed Contracts 부분 'Diary at.. 어쩌구' 이부분 복사 > contract_info.json 맨 아랫줄 address 붙여넣기 

*/


pragma solidity ^0.8.6;

contract Diary {
    // date => string
    address internal owner;     // 주인만 읽고 쓰도록 함
    mapping(uint256 => string) public diary;

    constructor() public {
        owner = msg.sender;     // 우리 지갑 
    }

    function getNumberlength (uint256 number) public pure returns(uint ){
        uint digits;
        while(number > 0){
            number /= 10;
            digits++;
        }
        return digits;
    }

    function addDiary(uint256 date, string memory content) public {
        require(msg.sender == owner,"Only owner of this contract can edit diary");
        require(getNumberlength(date) == 8,"Enter date in YYYYMMDD format");
        diary[date] = content;
    }

    function readDiary(uint256 date) public view returns (string memory) {
        require(msg.sender == owner, "Only owner of this contract can read diary");
        require(getNumberlength(date) == 8,"Enter date in YYYYMMDD format");
        return diary[date];
    }

}