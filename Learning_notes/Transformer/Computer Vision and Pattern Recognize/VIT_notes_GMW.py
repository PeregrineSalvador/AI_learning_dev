import torch
from torch import nn


# --- 组件 1: PatchEmbedding ---
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, dim):
        super().__init__()
        if not (image_size % patch_size == 0):
            raise ValueError("Image dimensions must be divisible by the patch size.")
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


# --- 组件 2: TransformerEncoderBlock ---
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_output, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        return x


# --- 主模型: VisionTransformer ---
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, num_classes,
                 dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, dim)
        num_patches = self.patch_embedding.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoderBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.patch_embedding(img)
        b, n, d = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        cls_token_output = x[:, 0]
        output = self.mlp_head(cls_token_output)
        return output


# --- 实例化并测试 ---

# CIFAR-10 实例参数
BATCH_SIZE = 4
IMAGE_SIZE = 32
IN_CHANNELS = 3
PATCH_SIZE = 4
NUM_CLASSES = 10
DIM = 512
DEPTH = 6
HEADS = 8
MLP_DIM = 2048

# 创建模型实例
vit_model = VisionTransformer(
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    in_channels=IN_CHANNELS,
    num_classes=NUM_CLASSES,
    dim=DIM,
    depth=DEPTH,
    heads=HEADS,
    mlp_dim=MLP_DIM
)

# 创建一个假的输入图像张量 (Batch, Channels, Height, Width)
dummy_img = torch.randn(BATCH_SIZE, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

# 将图像输入模型
logits = vit_model(dummy_img)

# 打印输出的形状
print(f"输入图像形状: {dummy_img.shape}")
print(f"模型输出 (Logits) 形状: {logits.shape}")

# 检查输出形状是否正确
assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
print("\n模型构建成功，输入输出形状正确！")