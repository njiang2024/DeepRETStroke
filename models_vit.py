from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, pre_train=False, progression=False,
                 combined = False, meta_nums=9, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.pre_train = pre_train
        self.progression = progression
        self.combined = combined
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            self.hidden_layer = torch.nn.Linear(embed_dim, embed_dim)
            self.head_2 = torch.nn.Linear(embed_dim, 2)
            self.head_5 = torch.nn.Linear(embed_dim, 5)
            self.meta_layer = torch.nn.Linear(meta_nums, 128)
            self.head_c2 = torch.nn.Linear(embed_dim+128, 2)
            self.head_c5 = torch.nn.Linear(embed_dim+128, 5)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward_head2(self, x, y=None, pre_logits: bool = False):
        if self.pre_train:
            x1 = self.head_2(self.hidden_layer(x))
            x2 = self.head_2(self.hidden_layer(x))
            x3 = self.head_2(self.hidden_layer(x))
            return x1, x2, x3
        else:
            if self.combined:
                y = self.meta_layer(y)
                y = torch.cat([x, y], dim=1)
                if self.progression:
                    return self.head_5(x), self.head_c5(y)
                else:
                    return self.head_2(x), self.head_c2(y)
            else:
                if self.progression:
                    return self.head_5(x)
                else:
                    return self.head_2(x)

    def forward(self, x):
        y = self.forward_features(x[0])
        if self.combined:
            y = self.forward_head2(y, x[1])
        else:
            y = self.forward_head2(y)
        return y

class VisionTransformer2(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=True, meta_nums=9, **kwargs):
        super(VisionTransformer2, self).__init__(**kwargs)

        self.global_pool = global_pool
        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']
        self.fc_norm = norm_layer(embed_dim)

        self.hidden_layer_0 = torch.nn.Linear(embed_dim, embed_dim)
        self.hidden_layer_1 = torch.nn.Linear(embed_dim, embed_dim)
        self.head_2_0 = torch.nn.Linear(embed_dim, 2)
        self.head_2_1 = torch.nn.Linear(embed_dim, 2)

        self.hidden_layer_c_0 = torch.nn.Linear(embed_dim+meta_nums, embed_dim+meta_nums)
        self.hidden_layer_c_1 = torch.nn.Linear(embed_dim+meta_nums, embed_dim+meta_nums)
        self.head_c2_0 = torch.nn.Linear(embed_dim+meta_nums, 2)
        self.head_c2_1 = torch.nn.Linear(embed_dim+meta_nums, 2)

        self.hidden_layer_t = torch.nn.Linear(embed_dim, embed_dim)
        self.head_2_t = torch.nn.Linear(embed_dim, 2)

        self.hidden_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.head_5 = torch.nn.Linear(embed_dim, 5)

        self.hidden_layer_c = torch.nn.Linear(embed_dim+meta_nums, embed_dim+meta_nums)
        self.head_c5 = torch.nn.Linear(embed_dim+meta_nums, 5)

        del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        outcome = self.fc_norm(x)

        return outcome

    def forward(self, x):
        photos, metadata, stage = x

        if stage == 1:
            y = self.forward_features(photos)
            y = self.head_5(self.hidden_layer(y))

            return y

        if stage == 2:
            if metadata[0] == 1:
                y0 = self.head_2_0(self.hidden_layer_0(photos[0]))
            else:
                y0 = 0
            if metadata[1] == 1:
                y1 = self.head_2_1(self.hidden_layer_1(photos[1]))
            else:
                y1 = 0

            return y0, y1

        if stage == 4:
            y = self.forward_features(photos)
            y0 = self.head_2_t(self.hidden_layer_t(y))
            y1 = self.head_5(self.hidden_layer(y))

            return y0, y1

        if stage == "fd":
            if metadata[0] == 1:
                y0 = self.head_c2_0(self.hidden_layer_c_0(photos[0]))
            else:
                y0 = 0
            if metadata[1] == 1:
                y1 = self.head_c2_1(self.hidden_layer_c_1(photos[1]))
            else:
                y1 = 0

            return y0, y1

        if stage == "fp":
            y = self.head_c5(self.hidden_layer_c(photos))

            return y

        if stage == "e":
            y = self.forward_features(photos)

            return  y

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        img_size=224,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patch16_2(**kwargs):
    model = VisionTransformer(
        img_size=256,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patch16_s(**kwargs):
    model = VisionTransformer2(
        img_size=256,patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == "__main__":
    model = vit_large_patch16_s()
    pt = torch.rand([2,3,256,256])
    ft = torch.rand([2,1024])
    fts = torch.rand([2,1024+9])
    packages = [fts, None, "fp"]
    a = model(packages)
    print(a)
