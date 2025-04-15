import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DirectionalProcessor(nn.Module):
    """ARC 그리드의 방향성 정보를 처리하는 모듈"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 8개 방향에 대한 가중치 (4개 기본 방향 + 4개 대각선 방향)
        self.direction_weights = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
            for _ in range(8)
        ])
        
        # 방향 정보를 처리하기 위한 추가 레이어
        self.direction_combine = nn.Linear(hidden_size * 8, hidden_size)
    
    def forward(self, grid_embedding):
        """
        그리드 임베딩의 방향성 정보 처리
        
        Args:
            grid_embedding: [batch, height, width, channels] 형태의 그리드 임베딩
        """
        batch, height, width, channels = grid_embedding.shape
        results = []
        
        # 8개 방향으로 처리
        # 0: 상, 1: 우상, 2: 우, 3: 우하, 4: 하, 5: 좌하, 6: 좌, 7: 좌상
        directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        
        for d, (dx, dy) in enumerate(directions):
            # 지정된 방향으로 그리드 시프트
            shifted = self._shift_grid(grid_embedding, dx, dy)
            
            # 방향별 가중치로 변환
            processed = torch.matmul(shifted.reshape(-1, channels), self.direction_weights[d])
            processed = processed.reshape(batch, height, width, -1)
            
            results.append(processed)
        
        # 모든 방향 결과 결합
        combined = torch.cat(results, dim=-1)
        return self.direction_combine(combined)
    
    def _shift_grid(self, grid, dx, dy):
        """
        그리드를 지정된 방향으로 시프트
        """
        batch, height, width, channels = grid.shape
        result = torch.zeros_like(grid)
        
        # 유효한 인덱스 계산
        valid_h_src = slice(max(0, -dy), min(height, height - dy))
        valid_w_src = slice(max(0, -dx), min(width, width - dx))
        valid_h_dst = slice(max(0, dy), min(height, height + dy))
        valid_w_dst = slice(max(0, dx), min(width, width + dx))
        
        # 시프트 적용
        result[:, valid_h_dst, valid_w_dst, :] = grid[:, valid_h_src, valid_w_src, :]
        
        return result


class CumMaxLayer(nn.Module):
    """CompressARC의 cummax 레이어와 유사한 기능 구현"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.projection = nn.Linear(channels * 2, channels)
    
    def forward(self, grid, mask=None):
        """
        그리드에 대해 누적 최대값 연산 수행
        
        Args:
            grid: [batch, height, width, channels] 형태의 그리드
            mask: 마스크 (선택 사항)
        """
        batch, height, width, channels = grid.shape
        result = torch.zeros_like(grid)
        
        # x 방향 누적 최대값
        x_cummax = torch.zeros_like(grid)
        for i in range(width):
            if i == 0:
                x_cummax[:, :, i, :] = grid[:, :, i, :]
            else:
                x_cummax[:, :, i, :] = torch.maximum(x_cummax[:, :, i-1, :], grid[:, :, i, :])
        
        # y 방향 누적 최대값
        y_cummax = torch.zeros_like(grid)
        for i in range(height):
            if i == 0:
                y_cummax[:, i, :, :] = grid[:, i, :, :]
            else:
                y_cummax[:, i, :, :] = torch.maximum(y_cummax[:, i-1, :, :], grid[:, i, :, :])
        
        # 결과 결합
        result = torch.cat([x_cummax, y_cummax], dim=-1)
        result = self.projection(result)
        
        # 마스크 적용 (있는 경우)
        if mask is not None:
            result = result * mask
        
        return result


class ShiftLayer(nn.Module):
    """CompressARC의 shift 레이어와 유사한 기능 구현"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.projection = nn.Linear(channels * 4, channels)
    
    def forward(self, grid, mask=None):
        """
        그리드에 대해 다양한 방향으로 시프트 연산 수행
        
        Args:
            grid: [batch, height, width, channels] 형태의 그리드
            mask: 마스크 (선택 사항)
        """
        batch, height, width, channels = grid.shape
        
        # 4개 방향 시프트 (상, 우, 하, 좌)
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        shifted_grids = []
        for dx, dy in directions:
            processor = DirectionalProcessor(self.channels)
            shifted = processor._shift_grid(grid, dx, dy)
            shifted_grids.append(shifted)
        
        # 모든 시프트 결과 결합
        combined = torch.cat(shifted_grids, dim=-1)
        result = self.projection(combined)
        
        # 마스크 적용 (있는 경우)
        if mask is not None:
            result = result * mask
        
        return result
