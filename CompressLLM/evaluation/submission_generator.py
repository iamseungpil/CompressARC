import json
import os
from tqdm import tqdm

def format_solution_for_kaggle(task_id, solution):
    """
    단일 태스크 솔루션을 Kaggle 제출 형식으로 변환
    """
    import numpy as np
    return {
        task_id: solution.tolist() if isinstance(solution, np.ndarray) else solution
    }

def generate_submission_file(solutions_dict, output_path="submission.json"):
    """
    모든 태스크의 솔루션을 Kaggle 제출 파일로 변환
    
    Args:
        solutions_dict: {task_id: solution_grid} 형식의 딕셔너리
        output_path: 저장할 파일 경로
    """
    submission = {}
    
    for task_id, solution in solutions_dict.items():
        submission.update(format_solution_for_kaggle(task_id, solution))
    
    # JSON 파일로 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(submission, f)
    
    print(f"Submission file created at {output_path}")
    return submission

def collect_results_from_directory(results_dir):
    """
    결과 디렉토리에서 모든 태스크의 최적 솔루션 수집
    
    Args:
        results_dir: 결과 저장 디렉토리
    """
    solutions_dict = {}
    
    # 각 태스크 폴더 탐색
    task_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    for task_id in tqdm(task_dirs, desc="Collecting solutions"):
        task_dir = os.path.join(results_dir, task_id)
        best_solution_path = os.path.join(task_dir, "best_solution.json")
        
        if os.path.exists(best_solution_path):
            with open(best_solution_path, 'r') as f:
                result = json.load(f)
                solutions_dict[task_id] = result["solution"]
    
    return solutions_dict
