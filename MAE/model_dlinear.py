import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes, remain_T


class MAE_Encoder_Dlinear(torch.nn.Module):
    def __init__(self,
                 sample_size=[2,240],
                 patch_size=(2,2),
                 emb_dim=192,  #192
                 num_layer=2, #12
                 num_head=3,
                 mask_ratio=0.75
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((sample_size[0] // patch_size[0]) * (sample_size[1] // patch_size[1]), 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(1, emb_dim, patch_size, patch_size)  #inchannel=1

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, x, backward_indexes=None, forward_indexes=None, remain_T=None):

        patches = self.patchify(x)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        if backward_indexes is None and remain_T is None and forward_indexes is None:
            patches, forward_indexes, backward_indexes, remain_T = self.shuffle(patches)
        else:
            patches = take_indexes(patches, forward_indexes)
            patches = patches[:remain_T]

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes, forward_indexes, remain_T

class MAE_Decoder_Dlinear(torch.nn.Module):
    def __init__(self,
                 sample_size= [2, 240],
                 patch_size=(2,2),
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((sample_size[0] // patch_size[0]) * (sample_size[1] // patch_size[1]) + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        """The output dimension of self.head is channel*patch_size[0] *patch_size[1]"""
        self.head = torch.nn.Linear(emb_dim, 1 * patch_size[0] *patch_size[1])  # 3 * patch_size ** 2
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)',
                        p1=patch_size[0], p2=patch_size[1], h=sample_size[0]//patch_size[0],
                                   w=sample_size[1]//patch_size[1])

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        x = self.patch2img(patches)
        mask = self.patch2img(mask)

        return x, mask

class MAE_ViT_Dlinear(torch.nn.Module):
    def __init__(self,
                 # image_size=32,
                 sample_shape=[2, 240],
                 patch_size=(2, 10),  # (2, 2)
                 emb_dim=64,  # 192
                 encoder_layer=2,  # 12
                 encoder_head=4,  # 3
                 decoder_layer=2,
                 decoder_head=4,  # 3
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder_avg = MAE_Encoder_Dlinear(sample_shape, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.encoder_err = MAE_Encoder_Dlinear(sample_shape, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder_Dlinear(sample_shape, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, x_avg, x_err):
        features_avg, backward_indexes, forward_indexes, remain_T = self.encoder_avg(x_avg)
        features_err, backward_indexes, forward_indexes, remain_T = self.encoder_err(x_err, backward_indexes, forward_indexes, remain_T)
        features = features_avg + features_err
        predicted_x, mask = self.decoder(features, backward_indexes)
        return predicted_x, mask


class ViT_Forecasting(torch.nn.Module):
    def __init__(self, encoder_avg : MAE_Encoder_Dlinear, encoder_err: MAE_Encoder_Dlinear, n_covariate=7, pred_len=24, n_sample=1) -> None:
        super().__init__()
        self.cls_token_avg = encoder_avg.cls_token
        self.cls_token_err = encoder_err.cls_token
        self.pos_embedding_avg = encoder_avg.pos_embedding
        self.pos_embedding_err = encoder_err.pos_embedding
        self.patchify_avg = encoder_avg.patchify
        self.patchify_err = encoder_err.patchify
        self.transformer_avg = encoder_avg.transformer
        self.transformer_err = encoder_err.transformer
        self.layer_norm_avg = encoder_avg.layer_norm
        self.layer_norm_err = encoder_err.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding_avg.shape[-1], n_covariate*pred_len*n_sample)

    def forward(self, x_avg, x_err):

        patches_avg = self.patchify_avg(x_avg)
        patches_err = self.patchify_err(x_err)
        patches_avg = rearrange(patches_avg, 'b c h w -> (h w) b c')
        patches_err = rearrange(patches_err, 'b c h w -> (h w) b c')
        patches_avg = patches_avg + self.pos_embedding_avg
        patches_err = patches_err + self.pos_embedding_err
        patches_avg = torch.cat([self.cls_token_avg.expand(-1, patches_avg.shape[1], -1), patches_err], dim=0)
        patches_err = torch.cat([self.cls_token_err.expand(-1, patches_err.shape[1], -1), patches_err], dim=0)
        patches_avg = rearrange(patches_avg, 't b c -> b t c')
        patches_err = rearrange(patches_err, 't b c -> b t c')
        features_avg = self.layer_norm_avg(self.transformer_avg(patches_avg))
        features_err = self.layer_norm_err(self.transformer_err(patches_err))
        features = features_avg + features_err
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits


if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 3, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    x = torch.rand(2, 1, 3, 200)
    encoder = MAE_Encoder_Dlinear()
    decoder = MAE_Decoder_Dlinear()
    features, backward_indexes = encoder(x)
    print(forward_indexes.shape)
    predicted_x, mask = decoder(features, backward_indexes)
    print(predicted_x.shape)
    loss = torch.mean((predicted_x - x) ** 2 * mask / 0.75)
    print(loss)