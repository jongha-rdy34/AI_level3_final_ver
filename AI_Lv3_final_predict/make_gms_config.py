"""gms_config.pkl 을 만듭니다. 주소·모델을 바꾸려면 아래 두 값을 고친 뒤 다시 실행하세요.

시험장에서는 이 파일이 미리 제공되며 절대 수정하면 안 됩니다(부정행위 규정). 연습용 재현입니다.
"""
import pickle
from pathlib import Path

from gms_client import GMSClient

BASE_URL = 'https://gms.ssafy.io/gmsapi/api.openai.com/v1'   # SSAFY GMS (OpenAI 호환)
MODEL = 'gpt-4.1-mini'

KEYS = ['접수일', '주문번호', '주문자', '연락처', '상품명', '상품분류', '주문사이즈', '주문색상', '수량',
        '결제금액_원', '요청유형', '사유유형', '교환사이즈', '교환색상', '반품배송비_원', '환불예정금액_원',
        '회수시도', '회수시군구', '회수도로명', '회수건물번호', '회수상세주소', '회수희망일']

if __name__ == '__main__':
    path = Path(__file__).resolve().parent / 'gms_config.pkl'
    with path.open('wb') as f:
        pickle.dump(GMSClient(BASE_URL, MODEL, mock_keys=KEYS), f)
    print('saved', path, '|', BASE_URL, '|', MODEL)
