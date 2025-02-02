import torch
from torch import nn 

class PatchEmbedding(nn.Module):
  def __init__(self,
               in_channels:int=3,
               patch_size:int=16,
               embedding_dim:int=768):
    super().__init__()

    self.patcher = nn.Conv2d(in_channels=in_channels,
                             out_channels=embedding_dim,
                             kernel_size=patch_size,
                             stride=patch_size,
                             padding=0)
    
    self.flatten = nn.Flatten(start_dim=2,
                              end_dim=3)
    
  def forward(self, x):
    x = self.patcher(x)
    x = self.flatten(x)
    return x.permute(0,2,1)   

class ViT(nn.Module):
  def __init__(self,
                img_size:int=224,
                in_channels:int=3,
                patch_size:int=16,
                num_transformers_layers:int=12,
                embedding_dim:int=768,
                mlp_size:int=3072,
                num_heads:int=12,
                mlp_dropout:float=0.1,
                embedding_dropout:float=0.1,
                num_classes:int=1000,):
    super().__init__()

    assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."
    
    self.num_patches = (img_size*img_size) // patch_size**2

    self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim), requires_grad=True)

    self.position_embedding = nn.Parameter(data=torch.randn(1,self.num_patches+1, embedding_dim), requires_grad=True)

    self.embedding_dropout = nn.Dropout(p=embedding_dropout)

    self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                          patch_size=patch_size,
                                          embedding_dim=embedding_dim)
    
    self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                    nhead=num_heads,
                                                    dropout=mlp_dropout,
                                                    dim_feedforward=mlp_size,
                                                    )

    self.transform_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                   num_layers=num_transformers_layers)
    
    self.classifer = nn.Sequential(
      nn.LayerNorm(normalized_shape=embedding_dim),
      nn.Linear(in_features=embedding_dim,
                out_features=num_classes)
    )

  def forward(self, x):
    BATCH_SIZE = x.shape[0]

    class_token = self.class_embedding.expand(BATCH_SIZE, -1, -1)

    x = self.patch_embedding(x)
    x = torch.cat((class_token, x), dim=1)
    x = self.position_embedding + x
    x = self.embedding_dropout(x)
    x = self.transform_encoder(x)
    x = self.classifer(x[:,0])
    return x
