# 소스 원본 : https://github.com/Kwangkee/Gachon/tree/main/practice/BCFL

'''
1. pip install web3 python-dotenv

2. https://mumbaifaucet.com/에서 예전 실습에서 생성한 메타마스크 지갑 주소 넣고 0.5 MATIC 받기
3.  Diary.sol, .env 파일 설명 참고하여 따라하기

4. main.py 실행
     1: read
     2: add
     * add 후 아무 내용 입력하고 read로 조회

5. 컨트랙트 조회는 : https://mumbai.polygonscan.com/tx/0x56fa9c80d394e6483f7af1af0ee7009333edda2be289ee7aa4b0af8194a91e9e
          조회는 컨트랙트 주소로 가능하며 contract_info.json 맨 밑의 address로 검색 가능
'''

from web3 import HTTPProvider, Web3
import json
from eth_account import Account
from dotenv import load_dotenv
import os

# Need for .env
load_dotenv()

contract_json_path = "contract_info.json"

# WARNING! Rpc url should be in .env
provider_address = os.environ.get("RPC_URL")
w3 = Web3(HTTPProvider(provider_address))

# WARNING! 지갑 개인 키는 절대 공개된 형태로 깃허브에 올리시면 안됩니다. 해킹 당해요!
user_account = os.environ.get("USER_ACCOUNT")
user_pk = os.environ.get("USER_PK")

with open(contract_json_path,"r") as json_file:
    contract_info = json.load(json_file)
    abi = contract_info['abi']
    contract_address = contract_info['contract_address']

print(abi)
# ABI, contract address is needed for Web3 contract instance
contract = w3.eth.contract(abi=abi, address=contract_address)

def get_tx_param():
    nonce = w3.eth.getTransactionCount(user_account)        # nonce: 위조 방지용
    tx_param = {         # transaction 파라미터 (가스비, 유저 등 정보 기재)
        "nonce":nonce,
        "gas":300000,
        "gasPrice":w3.toWei('50','gwei'),
        "from":user_account
    }
    return tx_param

def signedHash(tx, privatekey):
    signed_tx = Account.signTransaction(tx, privatekey)     # 서명 후 전송
    tx_hash = w3.eth.sendRawTransaction(signed_tx.rawTransaction)
    return tx_hash


def readDiary():    # contract 읽어옴
    date = input("Enter Diary's date : YYYYMMDD")
    date = int(date)
    diary = contract.functions.readDiary(date).call(get_tx_param())   # contract 사용하는 부분. 단순 view라서 가스비 안 듦
    print(diary)

def addDiary():     # contract 기록
    date = input("Enter Diary's data : YYYYMMDD")
    date = int(date)
    diary = input("Enter Diary's content")

    tx = contract.functions.addDiary(date,diary).buildTransaction(get_tx_param())   # 가스비 들여서 기록해야 함
    tx_hash = signedHash(tx,user_pk)
    print(tx_hash)

if __name__ == '__main__':
    while True:
        num = input("press 1 to read diary and 2 to Add diary")
        if num == "1":
            readDiary()
        else :
            addDiary()