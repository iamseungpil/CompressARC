import torch
import torch.nn.functional as F
import numpy as np

def compute_grid_matching_loss(output_grid, target_grid):
    """
    두 그리드 간의 일치도 손실 계산 (교차 엔트로피)
    
    Args:
        output_grid: 모델이 생성한 그리드
        target_grid: 실제 목표 그리드
    """
    # 그리드를 텐서로 변환
    if not isinstance(output_grid, torch.Tensor):
        output_grid = torch.tensor(output_grid, dtype=torch.long)
    if not isinstance(target_grid, torch.Tensor):
        target_grid = torch.tensor(target_grid, dtype=torch.long)
    
    # 교차 엔트로피 손실 (원-핫 인코딩 후)
    max_value = max(output_grid.max().item(), target_grid.max().item()) + 1
    output_onehot = F.one_hot(output_grid, num_classes=max_value).float()
    
    # 차원 재배열 (num_classes, height, width)
    output_onehot = output_onehot.permute(2, 0, 1)
    
    return F.cross_entropy(output_onehot, target_grid)

def compute_total_loss(reconstruction_loss, kl_divergence, beta=0.1):
    """
    재구성 손실과 KL divergence를 결합한 총 손실 계산
    
    Args:
        reconstruction_loss: 재구성 손실
        kl_divergence: KL divergence
        beta: KL divergence 가중치
    """
    return reconstruction_loss + beta * kl_divergence

def grid_accuracy(output_grid, target_grid):
    """
    생성된 그리드와 타겟 그리드 간의 정확도 계산
    
    Args:
        output_grid: 모델이 생성한 그리드
        target_grid: 실제 목표 그리드
    """
    if isinstance(output_grid, torch.Tensor):
        output_grid = output_grid.detach().cpu().numpy()
    if isinstance(target_grid, torch.Tensor):
        target_grid = target_grid.detach().cpu().numpy()
    
    # 그리드가 완전히 일치하는지 확인
    exact_match = np.array_equal(output_grid, target_grid)
    
    # 개별 셀 정확도 계산
    cell_accuracy = np.mean(output_grid == target_grid)
    
    return {
        'exact_match': exact_match,
        'cell_accuracy': float(cell_accuracy)
    }
