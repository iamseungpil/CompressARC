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


class MultiRandomNetwork(nn.Module):
    """
    다양한 차원 구성에 대한 랜덤 네트워크 모음
    """
    def __init__(self, dims_list, base_hidden_size=768):
        super().__init__()
        self.base_hidden_size = base_hidden_size
        self.dims_list = dims_list
        
        # 차원 구성별 랜덤 네트워크 생성
        self.networks = nn.ModuleDict()
        for dims in dims_list:
            dims_key = str(dims)  # 모듈 딕셔너리 키는 문자열이어야 함
            hidden_size = self.get_hidden_size(dims)
            self.networks[dims_key] = RandomNetwork(hidden_size)
    
    def get_hidden_size(self, dims):
        """차원 설정에 따른 은닉 크기 결정"""
        if dims[2] == 0:  # 방향 차원이 없는 경우
            return self.base_hidden_size
        else:  # 방향 차원이 있는 경우 더 작은 크기 사용
            return self.base_hidden_size // 2
    
    def forward_for_dims(self, dims, x):
        """특정 차원 구성에 대한 랜덤 네트워크 실행"""
        dims_key = str(dims)
        return self.networks[dims_key](x)
    
    def forward(self, x_dict):
        """모든 차원 구성에 대한 랜덤 네트워크 실행"""
        result = {}
        for dims_key, x in x_dict.items():
            str_key = str(dims_key)
            if str_key in self.networks:
                result[dims_key] = self.networks[str_key](x)
        return result
