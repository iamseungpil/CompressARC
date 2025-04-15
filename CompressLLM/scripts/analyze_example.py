import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np

from models.llama_arc_solver import LlamaARCSolver
from training.solution_tracker import SolutionTracker
from utils.grid_preprocessing import load_tasks
from utils.visualization import visualize_solution, visualize_grid
from evaluation.solution_evaluator import evaluate_solution

def analyze_example(task_id, split="training", output_dir=None, model_name=None, num_steps=30, iterations_per_step=50):
    """
    단일 ARC 태스크에 대한 상세 분석 실행 (latent 최적화 방식)
    
    Args:
        task_id: 분석할 태스크 ID
        split: 데이터 분할 (training/evaluation/test)
        output_dir: 결과 저장 경로
        model_name: 사용할 모델 이름
        num_steps: 훈련 스텝 수
        iterations_per_step: 각 스텝에서 latent 최적화 반복 횟수
    """
    # 태스크 로드
    tasks = load_tasks(split)
    task = next((t for t in tasks if t.task_id == task_id), None)
    
    if task is None:
        raise ValueError(f"Task {task_id} not found in {split} split")
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = f"results/{task_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 문제 시각화
    visualize_grid(task.test_input, f"{output_dir}/problem.png", title=f"Task {task_id} Input")
    
    # 샘플 예제 시각화
    for i, (input_grid, output_grid) in enumerate(task.examples[:-1]):
        visualize_grid(input_grid, f"{output_dir}/example_{i+1}_input.png", 
                     title=f"Example {i+1} Input")
        visualize_grid(output_grid, f"{output_dir}/example_{i+1}_output.png", 
                     title=f"Example {i+1} Output")
    
    # 모델 초기화
    model = LlamaARCSolver(model_name=model_name)
    
    # 솔루션 트래커 초기화
    tracker = SolutionTracker(task)
    
    # latent 벡터 초기화
    latent = model.init_latent()
    
    # 훈련 루프
    print(f"Training on task {task_id} for {num_steps} steps, {iterations_per_step} iterations per step")
    for step in tqdm(range(num_steps)):
        # 훈련 예제에서 입력/출력 쌍 가져오기
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
        
        # 시각화
        if (step + 1) % 5 == 0 or step == num_steps - 1:
            visualize_solution(tracker, f"{output_dir}/step_{step+1}")
            
            # 손실 곡선 시각화 (있는 경우)
            loss_history = tracker.metrics_history.get(step, {}).get('loss_history', None)
            if loss_history:
                plt.figure(figsize=(10, 6))
                plt.plot(loss_history)
                plt.title(f'Loss History at Step {step+1}')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.savefig(f"{output_dir}/loss_history_step_{step+1}.png")
                plt.close()
    
    # 최종 솔루션 생성
    print("Generating final solution with best latent")
    best_solution, best_step, best_latent = tracker.get_best_solution()
    
    # 최적 latent로 테스트 입력에 대한 솔루션 생성
    test_solution, kl_value, _ = model.solve(
        task.test_input, 
        examples=task.examples[:-1],
        latent=best_latent,
        num_iterations=100  # 테스트 시 더 많은 반복 수행
    )
    
    # 최종 솔루션 시각화
    visualize_grid(test_solution, f"{output_dir}/final_solution.png", title="Final Solution (KL: {:.4f})".format(kl_value))
    
    # 정답이 있으면 정확도 평가
    if task.test_output is not None:
        evaluation = evaluate_solution(test_solution, task.test_output)
        print(f"Solution accuracy: {evaluation}")
        visualize_grid(task.test_output, f"{output_dir}/groundtruth.png", title="Ground Truth")
        
        # 정확도 결과 저장
        with open(f"{output_dir}/evaluation.json", 'w') as f:
            import json
            json.dump(evaluation, f, indent=2)
    
    # latent 벡터 시각화 (PCA 또는 히트맵)
    if best_latent is not None:
        latent_np = best_latent.detach().cpu().numpy() if isinstance(best_latent, torch.Tensor) else np.array(best_latent)
        plt.figure(figsize=(10, 6))
        plt.imshow(latent_np.reshape(1, -1), aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Best Latent Vector Representation')
        plt.savefig(f"{output_dir}/best_latent.png")
        plt.close()
    
    print(f"Analysis completed. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze a single ARC task with latent optimization")
    parser.add_argument("--task-id", type=str, required=True, help="Task ID to analyze")
    parser.add_argument("--split", type=str, default="training", help="Data split (training/evaluation/test)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--model-name", type=str, default="barc0/Llama-3.1-ARC-Potpourri-Transduction-8B", help="Model name")
    parser.add_argument("--num-steps", type=int, default=30, help="Number of training steps")
    parser.add_argument("--iterations-per-step", type=int, default=50, help="Number of iterations per step")
    args = parser.parse_args()
    
    analyze_example(
        args.task_id, 
        split=args.split, 
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_steps=args.num_steps,
        iterations_per_step=args.iterations_per_step
    )

if __name__ == "__main__":
    main()
