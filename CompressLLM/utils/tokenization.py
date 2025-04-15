import torch
import numpy as np

def tokenize_grid(grid, tokenizer):
    """
    그리드를 토큰화
    
    Args:
        grid: ARC 그리드
        tokenizer: Llama 토크나이저
    """
    # 그리드를 문자열로 변환
    grid_str = "\n".join(" ".join(str(cell) for cell in row) for row in grid)
    
    # 토큰화
    return tokenizer(grid_str, return_tensors="pt").input_ids
    
def decode_grid_from_tokens(tokens, tokenizer):
    """
    토큰에서 그리드 디코딩
    
    Args:
        tokens: Llama 출력 토큰
        tokenizer: Llama 토크나이저
    """
    # 토큰을 텍스트로 디코딩
    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    
    # 텍스트에서 그리드 파싱
    try:
        lines = [line.strip() for line in text.strip().split("\n")]
        grid = []
        for line in lines:
            if not line or not any(c.isdigit() for c in line):
                continue
            row = []
            for cell in line.split():
                # 숫자만 추출
                digits = ''.join(c for c in cell if c.isdigit())
                if digits:
                    row.append(int(digits))
                else:
                    row.append(0)  # 숫자가 없으면 0으로 처리
            if row:  # 빈 행 무시
                grid.append(row)
                
        # 그리드가 비어있으면 기본값 반환
        if not grid:
            return [[0]]
            
        # 행 길이 맞추기 (가장 긴 행에 맞춤)
        max_len = max(len(row) for row in grid)
        for i in range(len(grid)):
            grid[i] = grid[i] + [0] * (max_len - len(grid[i]))
            
        return grid
    except Exception as e:
        print(f"Error decoding grid: {e}")
        print(f"Raw text: {text}")
        return [[0]]  # 오류 시 기본값 반환
