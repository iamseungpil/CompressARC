import torch
import torch.nn.functional as F
import numpy as np

def multitensor_attention(tensor_dict, query_dict):
    """
    다중 텐서 간 어텐션 매커니즘 구현
    
    Args:
        tensor_dict: 텐서 사전 {(dim1, dim2, ...): tensor, ...}
        query_dict: 쿼리 텐서 사전 {(dim1, dim2, ...): tensor, ...}
    """
    result = {}
    for key in tensor_dict:
        if key in query_dict:
            # 텐서와 쿼리
            tensor = tensor_dict[key]
            query = query_dict[key]
            
            # 어텐션 스코어 계산
            attention_score = torch.matmul(query, tensor.transpose(-2, -1)) / np.sqrt(tensor.shape[-1])
            attention_weights = F.softmax(attention_score, dim=-1)
            
            # 어텐션 가중 합
            attended_tensor = torch.matmul(attention_weights, tensor)
            result[key] = attended_tensor
        else:
            # 쿼리가 없으면 원본 텐서 유지
            result[key] = tensor_dict[key]
    
    return result

def multitensor_interpolate(tensor_dict, alpha=0.5):
    """
    다중 텐서 간 보간 연산
    
    Args:
        tensor_dict: 텐서 사전 {(dim1, dim2, ...): tensor, ...}
        alpha: 보간 계수 (0~1)
    """
    result = {}
    keys = list(tensor_dict.keys())
    
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            if i < j:  # 각 쌍에 대해 한 번만 계산
                # 두 텐서 가져오기
                tensor_i = tensor_dict[key_i]
                tensor_j = tensor_dict[key_j]
                
                # 형태 일치 여부 확인
                if tensor_i.shape == tensor_j.shape:
                    # 선형 보간
                    interpolated = alpha * tensor_i + (1 - alpha) * tensor_j
                    
                    # 새로운 키 생성 (두 키의 최대값으로)
                    new_key = tuple(max(a, b) for a, b in zip(key_i, key_j))
                    
                    # 결과에 추가
                    result[new_key] = interpolated
    
    # 원본 텐서도 포함
    for key in tensor_dict:
        if key not in result:
            result[key] = tensor_dict[key]
    
    return result
