import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiTensorDimension:
    """다중 텐서 차원 정의 - CompressARC와 유사한 차원 인덱싱 사용"""
    
    # 차원 의미: [예제, 색상, 방향, x축, y축]
    EXAMPLE = 0
    COLOR = 1
    DIRECTION = 2
    X_DIM = 3
    Y_DIM = 4

class MultiTensorLatent:
    """CompressARC의 multitensor_system과 유사한 다중 잠재 벡터 관리 시스템"""
    
    def __init__(self, base_hidden_size=768, device='cuda'):
        self.base_hidden_size = base_hidden_size
        self.device = device
        
        # 주요 텐서 조합 (CompressARC와 동일한 조합 사용)
        self.dims_list = [
            [1, 1, 0, 1, 1],  # 예제, 색상, 없음, x축, y축
            [1, 0, 0, 1, 1],  # 예제, 없음, 없음, x축, y축
            [1, 1, 1, 1, 1],  # 예제, 색상, 방향, x축, y축
            [1, 0, 1, 1, 1],  # 예제, 없음, 방향, x축, y축
            [1, 1, 0, 1, 0],  # 예제, 색상, 없음, x축, 없음
            [1, 1, 0, 0, 1],  # 예제, 색상, 없음, 없음, y축
        ]
        
        # 잠재 벡터 초기화
        self.latents = {}
        self.initialize_latents()
    
    def initialize_latents(self, std_dev=0.01):
        """모든 차원 조합에 대한 잠재 벡터 초기화"""
        for dims in self.dims_list:
            dims_key = tuple(dims)
            hidden_size = self.get_hidden_size(dims)
            
            # 랜덤 초기화된 latent 벡터 생성
            latent = torch.randn(1, hidden_size, device=self.device) * std_dev
            self.latents[dims_key] = nn.Parameter(latent, requires_grad=True)
    
    def get_hidden_size(self, dims):
        """차원 설정에 따른 은닉 크기 결정"""
        # CompressARC 방식과 유사하게 차원 구성에 따라 크기 조정
        if dims[2] == 0:  # 방향 차원이 없는 경우
            return self.base_hidden_size
        else:  # 방향 차원이 있는 경우 더 작은 크기 사용
            return self.base_hidden_size // 2
    
    def get_all_parameters(self):
        """최적화를 위한 모든 파라미터 반환"""
        return list(self.latents.values())
    
    def get_latent(self, dims):
        """특정 차원 구성에 해당하는 잠재 벡터 반환"""
        dims_key = tuple(dims)
        if dims_key in self.latents:
            return self.latents[dims_key]
        else:
            raise KeyError(f"잠재 벡터가 존재하지 않는 차원 구성: {dims}")
