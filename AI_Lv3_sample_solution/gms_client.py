"""P3용 언어모델 연결 객체 (공식 gms_config.pkl의 재현본).

공식 노트북은 `pickle.load(gms_config.pkl)`로 OpenAI 객체를 복원한 뒤
`client.api_key = USER_API_KEY` 를 넣고 `client.chat.completions.create(model=client.model, ...)` 로 부릅니다.
OpenAI 객체는 그대로 pickle 되지 않으므로, 같은 사용법을 가진 얇은 래퍼를 저장합니다.

- base_url / model 은 make_gms_config.py 에서 정합니다 (SSAFY GMS 기본값).
- api_key 가 'dummy' 또는 'mock' 이면 실제 호출 없이 '모든 값 null' JSON을 돌려주는 모의 모드입니다.
  (형식 검사 흐름만 확인용. 프롬프트 품질은 실제 키로 확인하세요.)
"""
import json
from types import SimpleNamespace


class _Completions:
    def __init__(self, owner):
        self._owner = owner

    def create(self, model, messages, **kwargs):
        owner = self._owner
        if owner.api_key in (None, '', 'dummy', 'mock'):
            keys = owner.mock_keys or []
            text = json.dumps({k: None for k in keys}, ensure_ascii=False)
            msg = SimpleNamespace(content=text, role='assistant')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason='stop')])
        from openai import OpenAI
        client = OpenAI(api_key=owner.api_key, base_url=owner.base_url)
        return client.chat.completions.create(model=model, messages=messages, **kwargs)


class GMSClient:
    def __init__(self, base_url, model, mock_keys=None):
        self.base_url = base_url
        self.model = model
        self.api_key = None
        self.mock_keys = mock_keys

    @property
    def chat(self):
        return SimpleNamespace(completions=_Completions(self))
