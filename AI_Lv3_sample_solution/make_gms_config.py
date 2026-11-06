"""gms_config.pkl 을 만듭니다. 주소·모델을 바꾸려면 아래 두 값을 고친 뒤 다시 실행하세요.

시험장에서는 이 파일이 미리 제공되며 절대 수정하면 안 됩니다(부정행위 규정). 연습용 재현입니다.
"""
import pickle
from pathlib import Path

from gms_client import GMSClient

BASE_URL = 'https://gms.ssafy.io/gmsapi/api.openai.com/v1'   # SSAFY GMS (OpenAI 호환)
MODEL = 'gpt-4.1-mini'

KEYS = ['임대인', '임차인', '중개인', '중개사무소', '시도', '시군구', '읍면', '도로명', '건물번호',
        '상세주소', '법정동', '건물명', '건물구조', '전용면적_m2', '전세보증금_원', '계약금_원',
        '중도금_원', '잔금_원', '계약체결일', '계약금지급일', '중도금지급일', '잔금지급일',
        '명도일', '입주일', '임대차시작일', '임대차종료일']

if __name__ == '__main__':
    path = Path(__file__).resolve().parent / 'gms_config.pkl'
    with path.open('wb') as f:
        pickle.dump(GMSClient(BASE_URL, MODEL, mock_keys=KEYS), f)
    print('saved', path, '|', BASE_URL, '|', MODEL)
