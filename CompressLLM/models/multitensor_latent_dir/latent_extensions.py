import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicLatentSystem(nn.Module):
    """
    동적 다중 텐서 잠재 벡터 시스템 - 
    기존 MultiTensorLatent를 확장하여 실행 시간에 차원 조합 구성 변경 가능
    """
    
    def __init__(self, base_hidden_size=768, device='cuda'):
        super().__init__()
        self.base_hidden_size = base_hidden_size
        self.device = device
        self.latents = nn.ParameterDict({})
        self.dims_registry = {}  # 현재 등록된 차원 조합 저장
    
    def register_dimension(self, dims, hidden_size=None):
        """
        새로운 차원 조합 등록 및 잠재 벡터 초기화
        
        Args:
            dims: 차원 구성 리스트 [예제, 색상, 방향, x축, y축]
            hidden_size: 사용자 지정 은닉 크기 (None이면 자동 계산)
        """
        dims_key = tuple(dims)
        if dims_key in self.dims_registry:
            return  # 이미 등록된 차원 조합
        
        # 은닉 크기 결정
        if hidden_size is None:
            if dims[2] == 1:  # 방향 차원이 있는 경우
                hidden_size = self.base_hidden_size // 2
            else:
                hidden_size = self.base_hidden_size
        
        # 차원 등록
        self.dims_registry[dims_key] = hidden_size
        
        # 잠재 벡터 초기화 (표준편차 0.01의 정규분포로)
        latent = torch.randn(1, hidden_size, device=self.device) * 0.01
        self.latents[str(dims_key)] = nn.Parameter(latent, requires_grad=True)
    
    def get_latent(self, dims):
        """
        특정 차원 구성에 해당하는 잠재 벡터 반환
        
        Args:
            dims: 차원 구성 리스트 또는 튜플
        """
        dims_key = tuple(dims) if isinstance(dims, list) else dims
        if dims_key in self.dims_registry:
            return self.latents[str(dims_key)]
        else:
            raise KeyError(f"등록되지 않은 차원 구성: {dims}")
    
    def mutate_latent(self, dims, mutation_scale=0.1):
        """
        기존 잠재 벡터에 변이 적용 (탐색 촉진)
        
        Args:
            dims: 차원 구성 리스트 또는 튜플
            mutation_scale: 변이 규모
        """
        dims_key = tuple(dims) if isinstance(dims, list) else dims
        if dims_key in self.dims_registry:
            # 현재 잠재 벡터에 가우시안 노이즈 추가
            noise = torch.randn_like(self.latents[str(dims_key)]) * mutation_scale
            self.latents[str(dims_key)].data += noise
    
    def reset_latent(self, dims, std_dev=0.01):
        """
        특정 잠재 벡터 재설정
        
        Args:
            dims: 차원 구성 리스트 또는 튜플
            std_dev: 초기화를 위한 표준편차
        """
        dims_key = tuple(dims) if isinstance(dims, list) else dims
        if dims_key in self.dims_registry:
            hidden_size = self.dims_registry[dims_key]
            latent = torch.randn(1, hidden_size, device=self.device) * std_dev
            self.latents[str(dims_key)].data = latent
    
    def get_all_parameters(self):
        """모든 잠재 벡터 파라미터 반환"""
        return list(self.latents.values())
    
    def get_all_latents(self):
        """모든 잠재 벡터를 딕셔너리로 반환"""
        return {eval(k): v for k, v in self.latents.items()}
