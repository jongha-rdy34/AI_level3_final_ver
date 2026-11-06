# Python 표준 라이브러리: json · re · ast · 자료형 검사

## json

### `json.loads(s)` → 파이썬 객체
- JSON의 `null/true/false` ↔ 파이썬 `None/True/False`. 객체 → `dict`, 배열 → `list`.
- 문법 오류면 `json.JSONDecodeError`(= `ValueError`의 하위 클래스).
- **JSON은 끝 쉼표(`[1, 2,]`)와 작은따옴표(`'a'`)를 허용하지 않습니다.**

### `json.dumps(obj, ensure_ascii=True, indent=None)` → 문자열
- `ensure_ascii=False`: 한글을 `\uXXXX`로 바꾸지 않고 그대로.

### JSON Lines (`.jsonl`)
- 한 줄에 JSON 객체 하나. 쓰기: `f.write(json.dumps(row, ensure_ascii=False) + "\n")`

## re

### `re.search(pattern, string)` → 첫 일치 `Match` 또는 `None`
- `m.group()` 일치 전체, `m.group(1)` 첫 괄호 그룹.
### `re.sub(pattern, repl, string)` → 바꾼 문자열
- `repl`에서 `\1`은 첫 그룹. 원시 문자열 `r"..."`을 쓰면 역슬래시가 그대로 전달됩니다.
### 자주 쓰는 패턴
- `r"-?\d+(\.\d+)?"`: 정수·소수(음수 포함)
- `r",\s*([}\]])"` → `r"\1"`: `}`나 `]` 바로 앞의 끝 쉼표 제거
- `r"\d\s*g\b"`: 숫자 뒤의 단위 g (`\b`는 단어 경계)

## str
- `s.find(sub)` / `s.rfind(sub)`: 처음 / 마지막 위치, 없으면 `-1`
- `s.strip()`, `s.lower()`, `s.startswith(prefix_or_tuple)`
- f-string에서 중괄호 문자 자체를 쓰려면 `{{`, `}}`

## ast

### `ast.literal_eval(s)`
- 파이썬 **리터럴**(숫자, 문자열, 튜플, 리스트, dict, `True/False/None`)만 안전하게 해석합니다. 함수 호출 등은 거절합니다. `{'a': None}` 같은 파이썬 표기 응답 복구에 씁니다.
- 실패하면 `ValueError` 또는 `SyntaxError`.

## 자료형 검사 함정

```python
isinstance(True, int)     # True  ← bool은 int의 하위 클래스
type(True) is int         # False
```
- 정수 필드 검사: `isinstance(v, int) and not isinstance(v, bool)`
- 숫자 필드 검사: `isinstance(v, (int, float)) and not isinstance(v, bool)`
- `round(x, 1)`: 소수 첫째 자리 반올림(파이썬은 짝수 쪽 반올림 규칙: `round(0.25, 1) == 0.2`)
