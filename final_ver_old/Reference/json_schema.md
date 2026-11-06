# JSON Schema (구조화 출력 설계)

LLM 출력을 프로그램이 바로 쓰려면 "어떤 키가, 어떤 자료형으로, 어떤 값만" 올 수 있는지 계약을 정해야 합니다. JSON Schema는 그 계약을 JSON(파이썬에서는 dict)으로 적는 표준입니다.

## 주요 키워드

| 키워드 | 뜻 | 예 |
|---|---|---|
| `type` | 자료형. 여러 개면 리스트 | `"integer"`, `"number"`, `"string"`, `"boolean"`, `"array"`, `"object"`, `"null"` / `["integer", "null"]` |
| `enum` | 허용값 목록. null을 허용하려면 `None` 포함 | `{"enum": ["M", "F", None]}` |
| `properties` | 객체의 키별 스키마 | `{"age": {...}, "sex": {...}}` |
| `required` | 반드시 있어야 하는 키 목록 | `["age", "sex"]` |
| `additionalProperties` | `False`면 `properties`에 없는 키 금지 | `False` |
| `items` | 배열 **원소 하나하나**의 스키마 | `{"type": "object", ...}` |
| `minimum` / `maximum` | 숫자 범위(포함) | `{"minimum": 0}` |
| `exclusiveMinimum` / `exclusiveMaximum` | 숫자 범위(미포함) | `{"exclusiveMinimum": 0}` |
| `minItems` / `maxItems` | 배열 길이 | |

- `"integer"`는 정수만, `"number"`는 정수와 실수 모두를 허용합니다. JSON Schema에서 `true`는 `integer`가 아닙니다(파이썬의 `isinstance(True, int)`와 다름).
- "모르면 null"을 허용하려면 `type`에 `"null"`을 함께 적습니다. 빠뜨리면 미확정 정보를 표현할 방법이 없어 LLM이 값을 추측하게 됩니다.

## 예

```python
schema = {
    "type": "object",
    "required": ["name", "count", "tags"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "count": {"type": ["integer", "null"], "minimum": 0},
        "tags": {"type": "array", "items": {"enum": ["a", "b"]}},
    },
}
```

## jsonschema 라이브러리 (설치되어 있을 때)

```python
import jsonschema
v = jsonschema.Draft202012Validator(schema)
errors = list(v.iter_errors(obj))      # 위반 목록 (비어 있으면 통과)
for e in errors:
    print(list(e.path), e.message)
jsonschema.validate(obj, schema)       # 위반이면 ValidationError 발생
```

## 프롬프트에 스키마를 넣을 때
- 키 이름·자료형·허용값·단위·"모르면 null" 규칙을 **명시**하고, 예시 출력 1개를 함께 줍니다.
- "JSON 객체 하나만, 설명·코드 블록 없이"를 지시해도 실제 응답은 어길 수 있으므로, 받는 쪽에서 **꺼내기 → 정규화 → 스키마 검사**를 항상 거칩니다.
