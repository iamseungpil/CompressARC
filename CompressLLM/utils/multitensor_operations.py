import torch
import torch.nn.functional as F
import numpy as np

def multitensor_normalize(tensor_dict, dim=-1):
    """
    다중 텐서 사전의 모든 텐서를 정규화
    
    Args:
        tensor_dict: 텐서 사전 {(dim1, dim2, ...): tensor, ...}
        dim: 정규화할 차원
    """
    result = {}
    for key, tensor in tensor_dict.items():
        mean = tensor.mean(dim=dim, keepdim=True)
        std = tensor.std(dim=dim, keepdim=True) + 1e-8
        result[key] = (tensor - mean) / std
    return result

def multitensor_apply(tensor_dict, fn):
    """
    다중 텐서 사전의 모든 텐서에 함수 적용
    
    Args:
        tensor_dict: 텐서 사전 {(dim1, dim2, ...): tensor, ...}
        fn: 적용할 함수
    """
    result = {}
    for key, tensor in tensor_dict.items():
        result[key] = fn(tensor)
    return result

def multitensor_kl_divergence(tensor_dict, reference_dict):
    """
    두 다중 텐서 사전 간의 KL 발산 계산
    
    Args:
        tensor_dict: 원본 텐서 사전
        reference_dict: 참조 텐서 사전
    """
    total_kl = 0
    kl_components = {}
    
    for key in tensor_dict:
        if key in reference_dict:
            # 두 텐서 가져오기
            tensor = tensor_dict[key]
            reference = reference_dict[key]
            
            # 소프트맥스 적용
            log_tensor = F.log_softmax(tensor, dim=-1)
            ref_softmax = F.softmax(reference, dim=-1)
            
            # KL 발산 계산
            kl = F.kl_div(log_tensor, ref_softmax, reduction='batchmean')
            
            total_kl += kl
            kl_components[key] = kl
    
    return total_kl, kl_components

def share_information(tensor_dict, direction='up'):
    """
    다중 텐서 간 정보 공유 (CompressARC의 share_up, share_down과 유사)
    
    Args:
        tensor_dict: 텐서 사전
        direction: 'up' 또는 'down'
    """
    result = {k: v.clone() for k, v in tensor_dict.items()}
    keys = list(tensor_dict.keys())
    
    if direction == 'up':
        # 하위 -> 상위 텐서 정보 공유
        for target_key in keys:
            for source_key in keys:
                if all(s <= t for s, t in zip(source_key, target_key)) and source_key != target_key:
                    # 차원 확장
                    source_tensor = tensor_dict[source_key]
                    for i, (s, t) in enumerate(zip(source_key, target_key)):
                        if s < t:
                            # 해당 차원 확장
                            source_tensor = source_tensor.unsqueeze(i).expand_as(result[target_key])
                    
                    # 정보 추가
                    result[target_key] = result[target_key] + source_tensor
    else:  # direction == 'down'
        # 상위 -> 하위 텐서 정보 공유
        for target_key in keys:
            for source_key in keys:
                if all(s >= t for s, t in zip(source_key, target_key)) and source_key != target_key:
                    # 차원 축소 (평균 계산)
                    source_tensor = tensor_dict[source_key]
                    for i, (s, t) in enumerate(zip(source_key, target_key)):
                        if s > t:
                            # 해당 차원 축소 (평균 계산)
                            source_tensor = source_tensor.mean(dim=i)
                    
                    # 정보 추가
                    result[target_key] = result[target_key] + source_tensor
    
    return result

def grid_to_multitensor(grid, dims_list):
    """
    그리드 데이터를 다중 텐서 형식으로 변환
    
    Args:
        grid: [batch, height, width, channels] 형태의 그리드
        dims_list: 변환할 차원 목록
    """
    batch, height, width, channels = grid.shape
    multitensor_dict = {}
    
    for dims in dims_list:
        if dims[3] == 1 and dims[4] == 1:  # x, y 차원이 모두 있는 경우
            multitensor_dict[tuple(dims)] = grid
        elif dims[3] == 1 and dims[4] == 0:  # x 차원만 있는 경우
            # y 차원에 대해 평균 계산
            multitensor_dict[tuple(dims)] = grid.mean(dim=2)
        elif dims[3] == 0 and dims[4] == 1:  # y 차원만 있는 경우
            # x 차원에 대해 평균 계산
            multitensor_dict[tuple(dims)] = grid.mean(dim=1)
    
    return multitensor_dict
