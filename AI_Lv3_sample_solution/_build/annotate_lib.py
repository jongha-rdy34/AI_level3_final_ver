"""코드 셀에 줄별 설명 주석을 붙이는 도구.
- 문자열(docstring, 프롬프트) 안의 줄, '\'로 이어지는 줄, 이미 주석이 있는 줄은 건드리지 않습니다.
- 설명은 줄 내용(앞뒤 공백 제거)을 키로 하는 사전에서 찾습니다."""
import io
import tokenize


def commentable_lines(src):
    """주석을 붙여도 되는 줄 번호(1부터) 집합."""
    bad = set()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return set()
    has_comment = set()
    for tok in tokens:
        if tok.type == tokenize.STRING or tok.type in (getattr(tokenize, 'FSTRING_START', -1), getattr(tokenize, 'FSTRING_MIDDLE', -1), getattr(tokenize, 'FSTRING_END', -1)):
            if tok.start[0] != tok.end[0]:
                bad.update(range(tok.start[0], tok.end[0] + 1))
        if tok.type == tokenize.COMMENT:
            has_comment.add(tok.start[0])
    # f-string 이 여러 줄에 걸친 경우(3.12): FSTRING_START~END 범위 전체 제외
    start = None
    for tok in tokens:
        if tok.type == getattr(tokenize, 'FSTRING_START', -1):
            start = tok.start[0]
        elif tok.type == getattr(tokenize, 'FSTRING_END', -1) and start is not None:
            if tok.end[0] != start:
                bad.update(range(start, tok.end[0] + 1))
            start = None
    lines = src.split('\n')
    ok = set()
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if not s or s.startswith('#') or i in bad or i in has_comment or s.endswith(chr(92)):
            continue
        ok.add(i)
    return ok


def annotate(src, comments, missing=None):
    ok = commentable_lines(src)
    out = []
    for i, line in enumerate(src.split('\n'), 1):
        key = line.strip()
        if i in ok:
            text = comments.get(key)
            if text:
                line = f'{line}  # {text}'
            elif missing is not None:
                missing.add(key)
        out.append(line)
    return '\n'.join(out)
