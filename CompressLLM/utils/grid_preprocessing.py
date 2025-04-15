import os
import json
import numpy as np

def load_tasks(split="training"):
    """
    ARC 데이터셋에서 태스크 로드
    
    Args:
        split: 데이터 분할 (training/evaluation/test)
    """
    # 데이터셋 경로
    dataset_dir = os.path.join("..", "dataset")
    
    # 태스크 파일 로드
    challenges_file = os.path.join(dataset_dir, f"arc-agi_{split}_challenges.json")
    solutions_file = os.path.join(dataset_dir, f"arc-agi_{split}_solutions.json")
    
    # 파일이 존재하지 않으면 에러
    if not os.path.exists(challenges_file):
        raise FileNotFoundError(f"Challenges file not found: {challenges_file}")
    
    # 챌린지 로드
    with open(challenges_file, 'r') as f:
        challenges = json.load(f)
    
    # 솔루션 로드 (테스트 분할에는 없을 수 있음)
    solutions = {}
    if os.path.exists(solutions_file):
        with open(solutions_file, 'r') as f:
            solutions = json.load(f)
    
    # 태스크 객체 생성
    tasks = []
    for task_id, challenge in challenges.items():
        task = ARCTask(
            task_id=task_id,
            examples=challenge,
            solution=solutions.get(task_id)
        )
        tasks.append(task)
    
    return tasks

def format_grid(grid):
    """
    ARC 그리드를 텍스트 형식으로 변환
    
    예: 
    [
      [0, 1, 0],
      [1, 1, 1],
      [0, 1, 0]
    ]
    
    ->
    
    "0 1 0
     1 1 1
     0 1 0"
    """
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

def format_examples(examples):
    """
    입출력 예제 쌍을 포맷팅
    
    Args:
        examples: [(input_grid, output_grid), ...] 형식의 예제 리스트
    """
    formatted = ""
    for i, (input_grid, output_grid) in enumerate(examples):
        formatted += f"Example {i+1}:\nInput:\n{format_grid(input_grid)}\nOutput:\n{format_grid(output_grid)}\n\n"
    
    return formatted

def parse_grid(text):
    """
    텍스트에서 그리드 파싱
    """
    lines = [line.strip() for line in text.strip().split("\n")]
    grid = []
    for line in lines:
        if not line:
            continue
        grid.append([int(cell) for cell in line.split()])
    return grid

class ARCTask:
    """ARC 태스크 정보 저장 클래스"""
    def __init__(self, task_id, examples, solution=None):
        self.task_id = task_id
        self.examples = []
        
        # 입출력 예제 쌍 처리
        for example in examples:
            input_grid = example["input"]
            output_grid = example["output"]
            self.examples.append((input_grid, output_grid))
        
        # 테스트 입력 (마지막 예제의 입력)
        self.test_input = examples[-1]["input"]
        
        # 정답 (테스트 출력)
        self.test_output = solution if solution is not None else None
