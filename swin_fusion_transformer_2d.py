import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    Wh, Ww = window_size
    x = x.view(B, H // Wh, Wh, W // Ww, Ww, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, Wh, Ww, C)
    return windows

def window_reverse(windows, window_size, H, W):
    
    Wh, Ww = window_size
    B = int(windows.shape[0] / (H * W / Wh / Ww))
    x = windows.view(B, H // Wh, W // Ww, Wh, Ww, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qk_scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.qk_scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww, Wh*Ww, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class PatchEmbedding(nn.Module):
    """2D Patch Embedding for time-series data."""
    def __init__(self, in_features, patch_size, embed_dim):
        super().__init__()
        # For time-series, patch_size can be (time_steps_per_patch, features_per_patch)
        # We'll flatten the patches and project them.
        self.patch_size = patch_size
        # Calculate the number of features after flattening a patch
        flattened_patch_features = patch_size[0] * patch_size[1]
        self.proj = nn.Linear(flattened_patch_features, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        # We need to reshape x into patches.
        # For simplicity, let's assume patch_size[0] divides seq_len and patch_size[1] divides num_features
        
        batch_size, seq_len, num_features = x.shape
        
        # Ensure dimensions are divisible by patch_size
        if seq_len % self.patch_size[0] != 0 or num_features % self.patch_size[1] != 0:
            raise ValueError("Input dimensions must be divisible by patch size.")
            
        num_patches_time = seq_len // self.patch_size[0]
        num_patches_features = num_features // self.patch_size[1]
        
        # Reshape to create patches
        x = x.reshape(batch_size, num_patches_time, self.patch_size[0], num_patches_features, self.patch_size[1])
        # Combine patch dimensions and flatten features within each patch
        x = x.permute(0, 1, 3, 2, 4).reshape(batch_size, num_patches_time * num_patches_features, -1)
        
        return self.proj(x)

class SwinTransformerBlock(nn.Module):
    """A single Swin Transformer Block without FiLM."""
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, cond_features=0, 
                 shift_size=0, img_size=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size

        assert 0 <= self.shift_size < self.window_size[0] or self.shift_size == 0, "shift_size must in 0-window_size[0]"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.img_size
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, fundamental_context, img_size):
        H, W = img_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        x = torch.nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self.attn_mask
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)

        shortcut = x
        x = self.norm2(x)
        x = shortcut + self.drop_path(self.mlp(x))

        return x

class SwinFusionTransformer2D(nn.Module):
    """Swin Fusion Transformer for 2D time-series data with late fusion of fundamental data."""
    def __init__(self, img_size=(30, 9), patch_size=(5, 9), in_chans=1, num_classes=3, embed_dim=96,
                 depths=[2, 2], num_heads=[3, 6], window_size=(6,1), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False,
                 patch_norm=True, fundamental_features_dim=10):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = self.embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.window_size = window_size

        self.patch_embed = PatchEmbedding(in_features=img_size[1], patch_size=patch_size, embed_dim=embed_dim)
        
        # MLP for fundamental data
        self.fundamental_mlp = nn.Sequential(
            nn.Linear(fundamental_features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Build Swin Transformer blocks
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            shift_size = 0 if i_layer % 2 == 0 else self.window_size[0] // 2
            self.layers.append(SwinTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                cond_features=0,  # FiLM is removed
                shift_size=shift_size,
                img_size=(img_size[0] // patch_size[0], img_size[1] // patch_size[1])
            ))

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # The head now takes the concatenated features
        self.head = nn.Linear(self.num_features + 32, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, fundamental_data):
        # x: (batch_size, seq_len, num_features)
        # fundamental_data: (batch_size, fundamental_features_dim)

        # 1. Process fundamental data with its MLP
        fundamental_output = self.fundamental_mlp(fundamental_data)

        # 2. Process technical/sentiment data with Swin Transformer
        x = self.patch_embed(x)
        H_feat = self.img_size[0] // self.patch_size[0]
        W_feat = self.img_size[1] // self.patch_size[1]
        
        for layer in self.layers:
            x = layer(x, None, (H_feat, W_feat)) # No fundamental_context passed to blocks

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)

        # 3. Concatenate the outputs
        combined_features = torch.cat((x, fundamental_output), dim=1)

        # 4. Final classification
        x = self.head(combined_features)

        return x

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # Dummy data
    batch_size = 4
    seq_len = 30
    num_features = 9 # Updated to 9 features after PCA
    
    fundamental_features_dim = 10 # 10 original fundamental features
    
    num_classes = 3 # Long, Short, Hold

    # Input time-series data (technical + sentiment) - now with 9 features
    dummy_x = torch.randn(batch_size, seq_len, num_features)
    # Input fundamental data
    dummy_fundamental = torch.randn(batch_size, fundamental_features_dim)

    # Model parameters
    embed_dim = 96
    patch_size = (5, 9) # Updated: 5 time steps, all 9 features per patch
    # Ensure img_size is compatible with patch_size for num_patches calculation
    img_size = (seq_len, num_features) # (30, 9)

    model = SwinFusionTransformer2D(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=1, # Not directly used for 2D time-series, but kept for Swin analogy
        num_classes=num_classes,
        embed_dim=embed_dim,
        depths=[2, 2], # Number of Swin Transformer blocks in each stage
        num_heads=[3, 6], # Number of attention heads for each stage
        window_size=(6,1), # Updated: Window size for Swin attention (H_feat, W_feat)
        fundamental_features_dim=fundamental_features_dim
    )

    output = model(dummy_x, dummy_fundamental)
    print(f"Output shape: {output.shape}") # Expected: (batch_size, num_classes)

    # Test with a different patch size that doesn't divide evenly (should raise error)
    try:
        model_invalid_patch = SwinFusionTransformer2D(img_size=img_size, patch_size=(4, 9), fundamental_features_dim=fundamental_features_dim)
        output_invalid = model_invalid_patch(dummy_x, dummy_fundamental)
    except ValueError as e:
        print(f"Caught expected error for invalid patch size: {e}")