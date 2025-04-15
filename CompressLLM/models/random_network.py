import torch
import torch.nn as nn

class RandomNetwork(nn.Module):
    """
    KL divergence 계산을 위한 고정된 랜덤 가중치 네트워크
    """
    def __init__(self, hidden_size=768):
        super().__init__()
        # 랜덤 가중치로 초기화된 네트워크
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 가중치 고정 (학습 안 됨)
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        return self.layers(x)
