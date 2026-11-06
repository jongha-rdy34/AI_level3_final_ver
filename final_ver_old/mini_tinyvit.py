"""연습용 작은 TinyViT (timm의 TinyVit과 같은 이름·API, 공식 구현 아님).

시험 지정 모델은 TinyViT-5M(입력 224, 약 5.4M 파라미터)입니다. 이 PC는 GPU·timm이 없어
같은 구조를 아주 작게 줄인 모델(입력 64, 약 0.2M 파라미터)로 연습합니다.

timm과 같게 맞춘 것
- 하위 모듈: patch_embed → stages[0](MBConv) → stages[1..3](downsample + attention blocks) → head
- head: global_pool → norm(LayerNorm) → fc(Linear)   (timm NormMlpClassifierHead와 같은 이름)
- model.pretrained_cfg, model.reset_classifier(n), model.get_classifier(),
  model.forward_features(x), model.forward_head(x, pre_logits=False), num_features
- create_model(name, pretrained=True, num_classes=None, drop_path_rate=0.0, checkpoint_path=None)
- resolve_data_config(model=model)
"""
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "assets" / "tiny_vit_practice.pt"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PRETRAIN_CLASSES = 16


class DropPath(nn.Module):
    """stochastic depth: 학습 중 잔차 가지를 표본마다 확률 p로 끈다."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class ConvNorm(nn.Module):
    def __init__(self, cin, cout, k=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class PatchEmbed(nn.Module):
    def __init__(self, cin, dim):
        super().__init__()
        self.conv1 = ConvNorm(cin, dim // 2, 3, 2)
        self.act = nn.GELU()
        self.conv2 = ConvNorm(dim // 2, dim, 3, 2)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


class MBConv(nn.Module):
    def __init__(self, dim, expand=4, drop_path=0.0):
        super().__init__()
        hid = dim * expand
        self.conv1 = ConvNorm(dim, hid, 1)
        self.act1 = nn.GELU()
        self.conv2 = ConvNorm(hid, hid, 3, groups=hid)
        self.act2 = nn.GELU()
        self.conv3 = ConvNorm(hid, dim, 1)
        self.act3 = nn.GELU()
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        y = self.conv3(self.act2(self.conv2(self.act1(self.conv1(x)))))
        return self.act3(x + self.drop_path(y))


class PatchMerging(nn.Module):
    """해상도를 1/2로 줄이고 채널을 늘린다."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = ConvNorm(cin, cout, 1)
        self.act1 = nn.GELU()
        self.conv2 = ConvNorm(cout, cout, 3, 2, groups=cout)
        self.act2 = nn.GELU()
        self.conv3 = ConvNorm(cout, cout, 1)

    def forward(self, x):
        return self.conv3(self.act2(self.conv2(self.act1(self.conv1(x)))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):                                   # x: (B, N, C)
        B, N, C = x.shape
        x = self.norm(x)
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = (attn.softmax(-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * ratio, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.norm(x))))


class TinyVitBlock(nn.Module):
    """attention → local depthwise conv → MLP (TinyViT 블록 구성)"""

    def __init__(self, dim, num_heads, drop_path=0.0):
        super().__init__()
        self.attn = Attention(dim, num_heads)
        self.local_conv = ConvNorm(dim, dim, 3, groups=dim)
        self.mlp = Mlp(dim)
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):                                   # x: (B, C, H, W)
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)                    # (B, HW, C)
        t = t + self.drop_path1(self.attn(t))
        x = self.local_conv(t.transpose(1, 2).reshape(B, C, H, W))
        t = x.flatten(2).transpose(1, 2)
        t = t + self.drop_path2(self.mlp(t))
        return t.transpose(1, 2).reshape(B, C, H, W)


class ConvLayer(nn.Module):
    def __init__(self, dim, depth, drop_paths):
        super().__init__()
        self.blocks = nn.Sequential(*[MBConv(dim, drop_path=drop_paths[i]) for i in range(depth)])

    def forward(self, x):
        return self.blocks(x)


class TinyVitStage(nn.Module):
    def __init__(self, cin, dim, depth, num_heads, drop_paths):
        super().__init__()
        self.downsample = PatchMerging(cin, dim)
        self.blocks = nn.Sequential(*[TinyVitBlock(dim, num_heads, drop_paths[i]) for i in range(depth)])

    def forward(self, x):
        return self.blocks(self.downsample(x))


class NormMlpClassifierHead(nn.Module):
    def __init__(self, dim, num_classes, drop_rate=0.0):
        super().__init__()
        self.in_features = dim
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(dim)
        self.flatten = nn.Flatten(1)
        self.pre_logits = nn.Identity()
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

    def reset(self, num_classes):
        self.fc = nn.Linear(self.in_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, pre_logits=False):
        x = self.flatten(self.global_pool(x))
        x = self.pre_logits(self.norm(x))
        return x if pre_logits else self.fc(self.drop(x))


class TinyVit(nn.Module):
    def __init__(self, num_classes=PRETRAIN_CLASSES, embed_dims=(24, 48, 64, 96), depths=(1, 1, 2, 1),
                 num_heads=(1, 2, 2, 3), drop_path_rate=0.0, drop_rate=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dims[-1]
        self.patch_embed = PatchEmbed(3, embed_dims[0])
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()   # 깊을수록 drop_path를 크게
        stages, c, start = [], embed_dims[0], 0
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            dp = dpr[start:start + depth]
            stages.append(ConvLayer(dim, depth, dp) if i == 0 else TinyVitStage(c, dim, depth, num_heads[i], dp))
            c, start = dim, start + depth
        self.stages = nn.Sequential(*stages)
        self.head = NormMlpClassifierHead(self.num_features, num_classes, drop_rate)
        self.pretrained_cfg = dict(
            architecture="tiny_vit_5m_224 (연습용 축소판)", input_size=(3, 64, 64), mean=IMAGENET_MEAN,
            std=IMAGENET_STD, interpolation="bicubic", crop_pct=0.95, num_classes=PRETRAIN_CLASSES,
            note="실제 tiny_vit_5m_224: input_size (3, 224, 224), 같은 mean/std",
        )

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head.reset(num_classes)

    def forward_features(self, x):
        return self.stages(self.patch_embed(x))

    def forward_head(self, x, pre_logits=False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def create_model(model_name="tiny_vit_5m_224", pretrained=True, num_classes=None, drop_path_rate=0.0,
                 drop_rate=0.0, checkpoint_path=None):
    """timm.create_model과 같은 모양의 생성 함수.

    pretrained=True이면 로컬 가중치(assets/tiny_vit_practice.pt)를 불러옵니다.
    num_classes를 주고 사전학습 클래스 수(16)와 다르면 head.fc를 새로 만듭니다(timm과 같은 동작).
    """
    if not model_name.startswith("tiny_vit"):
        raise ValueError(f"연습 세트에는 tiny_vit 계열만 있습니다: {model_name}")
    model = TinyVit(PRETRAIN_CLASSES, drop_path_rate=drop_path_rate, drop_rate=drop_rate)
    path = checkpoint_path or (WEIGHTS if pretrained else None)
    if path is not None:
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    if num_classes is not None and num_classes != PRETRAIN_CLASSES:
        model.reset_classifier(num_classes)
    return model


def resolve_data_config(args=None, model=None):
    """timm.data.resolve_data_config(model=model)처럼 전처리 설정 dict를 반환합니다."""
    cfg = model.pretrained_cfg
    return {k: cfg[k] for k in ("input_size", "interpolation", "mean", "std", "crop_pct")}
