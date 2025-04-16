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
        # 각 행을 줄바꾼으로 구분
        lines = [line.strip() for line in text.strip().split("\n")]
        
        # 그리드 데이터만 추출 (해당 영역 식별)
        grid_lines = []
        in_grid = False
        
        for line in lines:
            # 숫자가 포함된 행은 그리드 데이터로 간주
            if any(c.isdigit() for c in line):
                in_grid = True
                grid_lines.append(line)
            elif in_grid and not line:  # 그리드 데이터 이후의 빈 행을 만나면 데이터 종료
                break
        
        grid = []
        for line in grid_lines:
            row = []
            # 공백으로 구분된 셀 값 가져오기
            for cell in line.split():
                # 숫자만 추출
                digits = ''.join(c for c in cell if c.isdigit())
                if digits:
                    row.append(int(digits))
                else:
                    row.append(0)  # 숫자가 없으면 0으로 처리
            
            if row:  # 빈 행이 아닐 경우에만 추가
                grid.append(row)
        
        # 그리드가 비어있으면 기본값 반환
        if not grid:
            print("Warning: Empty grid detected. Returning default.")
            return [[0]]
        
        # 행 길이 맞추기 (가장 긴 행에 맞춤)
        max_len = max(len(row) for row in grid)
        for i in range(len(grid)):
            grid[i] = grid[i] + [0] * (max_len - len(grid[i]))
        
        # 그리드 유효성 추가 검사
        if any(not isinstance(cell, int) for row in grid for cell in row):
            print("Warning: Non-integer values in grid. Converting to integers.")
            grid = [[int(cell) if isinstance(cell, (int, float)) else 0 for cell in row] for row in grid]
        
        return grid
        
    except Exception as e:
        print(f"Error decoding grid: {e}")
        print(f"Raw text: {text}")
        return [[0]]  # 오류 시 기본값 반환
