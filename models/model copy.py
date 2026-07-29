import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------
# 3D Patch Embedding
# -------------------------------------------------------

class PatchEmbed3D(nn.Module):
    def _init_(
        self,
        in_channels=1,
        embed_dim=256,
        patch_size=(2, 8, 8)
    ):
        super()._init_()

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        """

        x = self.proj(x)

        B, E, T, H, W = x.shape

        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x


# -------------------------------------------------------
# Transformer Block
# -------------------------------------------------------

class STTransformerBlock(nn.Module):
    def _init_(
        self,
        dim,
        heads=8,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super()._init_()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        # Self Attention
        attn_out, _ = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x)
        )

        x = x + attn_out

        # Feed Forward
        x = x + self.mlp(self.norm2(x))

        return x


# -------------------------------------------------------
# REAL Spatiotemporal Transformer
# -------------------------------------------------------

class SpatiotemporalTransformer(nn.Module):

    def _init_(
        self,
        video_shape=(8, 60, 90),
        patch_size=(2, 6, 6),
        in_channels=1,
        embed_dim=256,
        depth=6,
        heads=8,
        num_classes=3,
        dropout=0.1
    ):
        super()._init_()

        T, H, W = video_shape
        pt, ph, pw = patch_size

        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        num_patches = (
            (T // pt) *
            (H // ph) *
            (W // pw)
        )

        # CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.blocks = nn.ModuleList([
            STTransformerBlock(
                dim=embed_dim,
                heads=heads,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Metadata fusion
        self.meta_fc = nn.Linear(9, 32)

        # Final prediction
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 32, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, X):

        """
        X[0] = video tensor
               shape: (B, C, T, H, W)

        X[1] = velocity metadata
        X[2] = quaternion metadata
        """

        video = X[0]
        vel = X[1]
        quat = X[2]

        B = video.shape[0]

        # -----------------------------------------
        # Patch Embedding
        # -----------------------------------------

        x = self.patch_embed(video)

        # -----------------------------------------
        # Add CLS token
        # -----------------------------------------

        cls = self.cls_token.expand(B, -1, -1)

        x = torch.cat([cls, x], dim=1)

        # -----------------------------------------
        # Positional Encoding
        # -----------------------------------------

        x = x + self.pos_embed

        x = self.dropout(x)

        # -----------------------------------------
        # Transformer Encoder
        # -----------------------------------------

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # CLS token output
        x = x[:, 0]

        # -----------------------------------------
        # Metadata Fusion
        # -----------------------------------------

        meta = torch.cat([vel * 0.1, quat], dim=1)

        meta = self.meta_fc(meta)

        # -----------------------------------------
        # Final Prediction
        # -----------------------------------------

        x = torch.cat([x, meta], dim=1)

        out = self.head(x)

        return out


# -------------------------------------------------------
# Example Usage
# -------------------------------------------------------

if _name_ == "_main_":

    model = SpatiotemporalTransformer()

    dummy_video = torch.randn(
        2,      # batch
        1,      # channels
        8,      # time
        60,     # height
        90      # width
    )

    vel = torch.randn(2, 5)
    quat = torch.randn(2, 4)

    output = model([dummy_video, vel, quat])

    print(output.shape)