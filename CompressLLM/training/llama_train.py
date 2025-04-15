import torch
from tqdm import tqdm
import os

from models.llama_arc_solver import LlamaARCSolver
from training.solution_tracker import SolutionTracker
from utils.visualization import visualize_solution

def train_model(task, output_dir, model_name=None, num_steps=1500):
    """
    단일 태스크에 대해 Llama 모델 훈련
    
    Args:
        task: ARC 태스크 객체
        output_dir: 결과 저장 경로
        model_name: 사용할 모델 이름 (기본값: Llama-3.1-ARC)
        num_steps: 훈련 스텝 수
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 초기화
    model = LlamaARCSolver(model_name=model_name)
    
    # 파인튜닝할 파라미터만 선택 (인코더 projection 레이어)
    optimizer = torch.optim.Adam(
        [p for n, p in model.named_parameters() if 'encoder_projector' in n],
        lr=1e-4
    )
    
    # 솔루션 트래커 초기화
    tracker = SolutionTracker(task)
    
    # 훈련 루프
    for step in tqdm(range(num_steps)):
        # 태스크의 훈련 예제에서 입력/출력 쌍 가져오기
        for input_grid, target_grid in task.examples:
            # 훈련 단계 수행
            solution, metrics = model.train_step(
                input_grid, 
                target_grid, 
                examples=task.examples[:-1],  # 현재 예제 제외한 다른 예제들
                optimizer=optimizer
            )
            
            # 솔루션 및 메트릭 로깅
            tracker.log_solution(step, solution, metrics)
        
        # 일정 간격으로 시각화
        if (step + 1) % 50 == 0:
            visualize_solution(tracker, f"{output_dir}/step_{step+1}")
    
    # 최적 솔루션 및 메트릭 저장
    best_solution, best_step = tracker.get_best_solution()
    save_results(best_solution, tracker.metrics_history[best_step], f"{output_dir}/best_solution.json")
    
    # KL 및 재구성 오류 커브 저장
    metrics_history = tracker.get_metrics_history()
    save_metrics(metrics_history, f"{output_dir}/metrics.npz")
    
    return model, tracker

def save_results(solution, metrics, output_path):
    """결과 저장 유틸리티"""
    import json
    result = {
        "solution": solution if isinstance(solution, list) else solution.tolist(),
        "metrics": {k: float(v) for k, v in metrics.items()}
    }
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

def save_metrics(metrics_history, output_path):
    """메트릭 히스토리 저장 (numpy 형식)"""
    import numpy as np
    np.savez(
        output_path, 
        reconstruction_error=np.array(metrics_history['reconstruction_error']),
        kl_curves={k: np.array(v) for k, v in metrics_history['kl_curves'].items()}
    )
