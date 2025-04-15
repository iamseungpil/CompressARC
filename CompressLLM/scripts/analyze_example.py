import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from models.llama_arc_solver import LlamaARCSolver
from training.solution_tracker import SolutionTracker
from utils.grid_preprocessing import load_tasks
from utils.visualization import visualize_solution, visualize_grid
from evaluation.solution_evaluator import evaluate_solution

def analyze_example(task_id, split="training", output_dir=None, model_name=None, num_steps=1500):
    """
    단일 ARC 태스크에 대한 상세 분석 실행
    
    Args:
        task_id: 분석할 태스크 ID
        split: 데이터 분할 (training/evaluation/test)
        output_dir: 결과 저장 경로
        model_name: 사용할 모델 이름
        num_steps: 훈련 스텝 수
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
    
    # 파인튜닝할 파라미터만 선택 (인코더 projection 레이어)
    optimizer = torch.optim.Adam(
        [p for n, p in model.named_parameters() if 'encoder_projector' in n],
        lr=1e-4
    )
    
    # 솔루션 트래커 초기화
    tracker = SolutionTracker(task)
    
    # 훈련 루프
    print(f"Training on task {task_id} for {num_steps} steps")
    for step in tqdm(range(num_steps)):
        # 훈련 예제에서 입력/출력 쌍 가져오기
        for input_grid, target_grid in task.examples[:-1]:  # 마지막 예제는 테스트용
            # 훈련 단계 수행
            solution, metrics = model.train_step(
                input_grid, 
                target_grid, 
                examples=[ex for ex in task.examples[:-1] if ex[0] != input_grid],  # 현재 예제 제외한 다른 예제들
                optimizer=optimizer
            )
            
            # 솔루션 및 메트릭 로깅
            tracker.log_solution(step, solution, metrics)
        
        # 일정 간격으로 시각화
        if (step + 1) % 50 == 0:
            visualize_solution(tracker, f"{output_dir}/step_{step+1}")
    
    # 최종 솔루션 생성
    print("Generating final solution")
    solution, _ = model.solve(task.test_input, examples=task.examples[:-1])
    
    # 최종 솔루션 시각화
    visualize_grid(solution, f"{output_dir}/final_solution.png", title="Final Solution")
    
    # 정답이 있으면 정확도 평가
    if task.test_output is not None:
        evaluation = evaluate_solution(solution, task.test_output)
        print(f"Solution accuracy: {evaluation}")
        visualize_grid(task.test_output, f"{output_dir}/groundtruth.png", title="Ground Truth")
    
    print(f"Analysis completed. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze a single ARC task")
    parser.add_argument("--task-id", type=str, required=True, help="Task ID to analyze")
    parser.add_argument("--split", type=str, default="training", help="Data split (training/evaluation/test)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--model-name", type=str, default="barc0/Llama-3.1-ARC-Potpourri-Transduction-8B", help="Model name")
    parser.add_argument("--num-steps", type=int, default=1500, help="Number of training steps")
    args = parser.parse_args()
    
    analyze_example(
        args.task_id, 
        split=args.split, 
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_steps=args.num_steps
    )

if __name__ == "__main__":
    main()
