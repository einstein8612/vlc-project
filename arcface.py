import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, margin=0.5, scale=30.0, device='cpu'):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, embedding_dim)).to(device)
        nn.init.xavier_uniform_(self.W)
        self.margin = margin
        self.scale = scale
        self.device = device
    
    def forward(self, embeddings, labels):
        # Normalize weights + embeddings
        embeddings = F.normalize(embeddings, dim=1)
        W = F.normalize(self.W, dim=1)

        # Cosine similarity
        cos_theta = torch.matmul(embeddings, W.t())  # [B, C]
        theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

        # Add margin to correct class
        target_logits = torch.cos(theta + self.margin)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        # Replace only the target logits
        logits = cos_theta * (1 - one_hot) + target_logits * one_hot

        # Apply scale and CE loss
        logits *= self.scale
        loss = F.cross_entropy(logits, labels)

        return loss
