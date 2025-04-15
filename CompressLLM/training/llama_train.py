import torch
from tqdm import tqdm
import os
import numpy as np

from models.llama_arc_solver import LlamaARCSolver
from training.solution_tracker import SolutionTracker
from utils.visualization import visualize_solution

def train_model(task, output_dir, model_name=None, num_steps=1500, iterations_per_step=20):
    """
    단일 태스크에 대해 Llama 모델 훈련 (latent 최적화 방식)
    
    Args:
        task: ARC 태스크 객체
        output_dir: 결과 저장 경로
        model_name: 사용할 모델 이름 (기본값: Llama-3.1-ARC)
        num_steps: 전체 훈련 스텝 수
        iterations_per_step: 각 스텝에서 latent 최적화 반복 횟수
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 모델 초기화
    model = LlamaARCSolver(model_name=model_name)
    
    # 솔루션 트래커 초기화
    tracker = SolutionTracker(task)
    
    # latent 벡터 초기화
    latent = model.init_latent()
    
    # 훈련 루프
    for step in tqdm(range(num_steps)):
        # 태스크의 훈련 예제에서 입력/출력 쌍 가져오기
        for input_grid, target_grid in task.examples[:-1]:  # 마지막 예제는 테스트용
            # latent 최적화
            solution, metrics, latent = model.optimize_latent(
                input_grid, 
                target_grid, 
                examples=[ex for ex in task.examples[:-1] if ex[0] != input_grid],  # 현재 예제 제외한 다른 예제들
                latent=latent,
                optimizer=torch.optim.Adam([latent], lr=0.01),
                num_iterations=iterations_per_step
            )
            
            # 솔루션 및 메트릭 로깅
            tracker.log_solution(step, solution, metrics, latent)
        
        # 일정 간격으로 시각화
        if (step + 1) % 50 == 0:
            visualize_solution(tracker, f"{output_dir}/step_{step+1}")
    
    # 최적 솔루션 및 메트릭 저장
    best_solution, best_step, best_latent = tracker.get_best_solution()
    save_results(best_solution, tracker.metrics_history[best_step], best_latent, f"{output_dir}/best_solution.json")
    
    # KL 및 재구성 오류 커브 저장
    metrics_history = tracker.get_metrics_history()
    save_metrics(metrics_history, f"{output_dir}/metrics.npz")
    
    # 테스트 입력에 대한 솔루션 생성
    test_solution, _, _ = model.solve(
        task.test_input, 
        examples=task.examples[:-1],
        latent=best_latent,
        num_iterations=100  # 테스트 시 더 많은 반복 수행
    )
    
    # 테스트 솔루션 저장
    save_test_solution(test_solution, f"{output_dir}/test_solution.json")
    
    return model, tracker, best_latent

def save_results(solution, metrics, latent, output_path):
    """결과 저장 유틸리티"""
    import json
    result = {
        "solution": solution if isinstance(solution, list) else solution.tolist(),
        "metrics": {k: float(v) if not isinstance(v, list) else v for k, v in metrics.items() if k != 'loss_history'}
    }
    
    # latent 벡터 저장 (numpy로 변환하여 저장)
    if latent is not None:
        if isinstance(latent, torch.nn.Parameter) or isinstance(latent, torch.Tensor):
            result["latent"] = latent.detach().cpu().numpy().tolist()
        else:
            result["latent"] = latent
    
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

def save_test_solution(solution, output_path):
    """테스트 솔루션 저장 유틸리티"""
    import json
    result = {
        "solution": solution if isinstance(solution, list) else solution.tolist()
    }
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
