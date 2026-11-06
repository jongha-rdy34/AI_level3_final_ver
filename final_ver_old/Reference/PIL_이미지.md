# Pillow (PIL) — 이미지 파일 다루기

```python
from PIL import Image
```

### `Image.open(fp)` → `Image`
- **파일 머리만 읽습니다(지연 로딩).** 잘린 파일·빈 파일은 `open`이 아니라 픽셀을 실제로 읽는 `im.load()`, `convert`, `resize` 등에서 오류가 납니다. 파일 점검은 `im.load()`까지 해 보세요.
- 빈 파일·이미지가 아닌 파일: `PIL.UnidentifiedImageError` / 잘린 파일: `OSError: image file is truncated`
- `with Image.open(p) as im:` 형태로 쓰면 파일 핸들이 닫힙니다(블록 밖에서 쓰려면 안에서 `load()`·`convert()`).

### 속성
- `im.mode`: `"RGB"`(3채널), `"RGBA"`(투명도 포함), `"L"`(흑백 1채널), `"LA"`, `"P"`(팔레트), `"CMYK"`(인쇄용 4채널, JPEG에 흔함)
- `im.size` → `(width, height)` (numpy 배열 모양은 `(height, width, C)` — 순서가 반대)
- `im.info`: 부가 정보 dict. 팔레트 이미지의 투명색은 `"transparency"` 키

### `im.convert(mode)`
- `"RGB"`로 바꾸면 L·CMYK·P도 3채널이 됩니다.
- **RGBA → RGB는 알파를 그냥 버립니다.** 투명 픽셀에 저장된 RGB 값(흔히 0,0,0 = 검정)이 그대로 드러나 배경이 검게 될 수 있습니다. 흰 배경이 필요하면 합성합니다:

```python
im = im.convert("RGBA")
bg = Image.new("RGB", im.size, (255, 255, 255))
bg.paste(im, mask=im.getchannel("A"))     # 또는 Image.alpha_composite(흰 RGBA, im).convert("RGB")
```

### `Image.new(mode, size, color)` / `im.paste(im2, box=(x, y), mask=None)`
- `paste`의 `mask`: 같은 크기의 L 또는 RGBA 이미지. 0인 곳은 붙이지 않음.
- 비율을 지킨 정사각형 패딩: 긴 변 크기의 캔버스를 만들고 `((s - w) // 2, (s - h) // 2)`에 붙임.

### `im.resize((w, h), resample)`
- `Image.BICUBIC`, `Image.BILINEAR`, `Image.NEAREST`, `Image.LANCZOS`. 사전학습 설정의 `interpolation`(TinyViT는 bicubic)에 맞추면 좋습니다.
- 가로세로 비율이 다른 사진을 바로 정사각형으로 `resize`하면 물체가 늘어납니다(긴바지가 짧아 보이는 등).

### 그 밖에
- `im.getbbox()`: 0이 아닌 영역의 상자 (투명 배경 물체의 경계)
- `im.transpose(Image.FLIP_LEFT_RIGHT)`, `im.rotate(deg, expand=True, fillcolor=...)`
- `np.asarray(im)` → `(H, W, C)` uint8, `Image.fromarray(arr)` ← uint8 배열
- 파일 바이트가 같은지(완전 중복): `hashlib.md5(Path(p).read_bytes()).hexdigest()`. 크기만 다른 비슷한 사진은 잡지 못합니다(그럴 땐 축소 이미지 해시 등).
