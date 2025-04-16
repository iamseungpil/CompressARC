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
            target_tensor = result[target_key]
            
            for source_key in keys:
                # 상위 텐서로의 공유 조건: 모든 차원이 같거나 작아야 함
                if source_key != target_key and all(s <= t for s, t in zip(source_key, target_key)):
                    try:
                        source_tensor = tensor_dict[source_key].clone()
                        source_shape = list(source_tensor.shape)
                        target_shape = list(target_tensor.shape)
                        
                        # 여기서 실제 텐서 형태와 차원 구성의 관계를 확인
                        if len(source_shape) != len(target_shape):
                            print(f"Warning: Dimension mismatch between {source_key}:{source_shape} and {target_key}:{target_shape}")
                            continue
                        
                        # 확장이 필요한 차원 찾기
                        reshape_needed = False
                        view_shape = list(source_shape)
                        expand_shape = list(target_shape)
                        
                        # 각 차원별 확장 확인
                        for i, (s_dim, t_dim) in enumerate(zip(source_shape, target_shape)):
                            if s_dim != t_dim:
                                reshape_needed = True
                                if s_dim == 1:  # 싱글톤 차원은 확장 가능
                                    # 그대로 유지
                                    pass
                                else:
                                    # 차원이 다르고 싱글톤도 아니면 확장 불가
                                    print(f"Cannot expand dimension {i} from {s_dim} to {t_dim}")
                                    reshape_needed = False
                                    break
                        
                        # 실제 차원 확장 수행
                        if reshape_needed:
                            # 확장 작업 시도
                            try:
                                source_tensor = source_tensor.expand(expand_shape)
                                result[target_key] = result[target_key] + source_tensor
                            except RuntimeError as e:
                                print(f"Expansion error: {e}")
                                # 차원별 점진적 확장 시도
                                try:
                                    for i, (s_dim, t_dim) in enumerate(zip(source_shape, target_shape)):
                                        if s_dim != t_dim and s_dim == 1:
                                            expand_dims = [t_dim if j == i else source_shape[j] for j in range(len(source_shape))]
                                            source_tensor = source_tensor.expand(expand_dims)
                                    result[target_key] = result[target_key] + source_tensor
                                except Exception as e2:
                                    print(f"Progressive expansion failed: {e2}")
                        else:
                            # 확장이 필요없거나 불가능한 경우, 가능하면 그대로 더함
                            if source_shape == target_shape:
                                result[target_key] = result[target_key] + source_tensor
                    
                    except Exception as e:
                        print(f"Error in share_up for {source_key} -> {target_key}: {e}")
    
    else:  # direction == 'down'
        # 상위 -> 하위 텐서 정보 공유
        for target_key in keys:
            target_tensor = result[target_key]
            
            for source_key in keys:
                # 하위 텐서로의 공유 조건: 모든 차원이 같거나 커야 함
                if source_key != target_key and all(s >= t for s, t in zip(source_key, target_key)):
                    try:
                        source_tensor = tensor_dict[source_key].clone()
                        source_shape = list(source_tensor.shape)
                        target_shape = list(target_tensor.shape)
                        
                        # 텐서 형태 불일치 확인
                        if len(source_shape) != len(target_shape):
                            print(f"Warning: Dimension mismatch between {source_key}:{source_shape} and {target_key}:{target_shape}")
                            continue
                        
                        # 축소 필요한 차원 확인
                        reduce_dims = []
                        for i, (s_dim, t_dim) in enumerate(zip(source_shape, target_shape)):
                            if s_dim > t_dim:  # 축소 필요
                                reduce_dims.append(i)
                        
                        # 각 축소 차원에 대해 평균 계산 (역순으로 처리)
                        reduced_tensor = source_tensor
                        for dim in sorted(reduce_dims, reverse=True):
                            # keepdim=True로 차원 유지하면서 평균 계산
                            reduced_tensor = reduced_tensor.mean(dim=dim, keepdim=True)
                        
                        # 불필요한 차원 제거 (squeeze)
                        for dim in sorted(reduce_dims, reverse=True):
                            if reduced_tensor.shape[dim] == 1:
                                reduced_tensor = reduced_tensor.squeeze(dim)
                        
                        # 텐서 형태 확인 후 더하기
                        if list(reduced_tensor.shape) == target_shape:
                            result[target_key] = result[target_key] + reduced_tensor
                        else:
                            print(f"Shape mismatch after reduction: {reduced_tensor.shape} vs {target_shape}")
                            
                    except Exception as e:
                        print(f"Error in share_down for {source_key} -> {target_key}: {e}")
    
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
