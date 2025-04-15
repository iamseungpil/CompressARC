import os
import argparse
import multiprocessing
from tqdm import tqdm
import torch

from models.llama_arc_solver import LlamaARCSolver
from training.solution_tracker import SolutionTracker
from utils.grid_preprocessing import load_tasks
from evaluation.submission_generator import generate_submission_file
from utils.visualization import visualize_solution

def run_single_task(task, output_dir, model_name=None, num_steps=1500, device=None):
    """단일 태스크에 대한 훈련 및 평가 실행"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미 처리된 태스크인지 확인
    if os.path.exists(os.path.join(output_dir, "best_solution.json")):
        print(f"Task {task.task_id} already processed. Skipping.")
        return None
    
    # 모델 초기화
    model = LlamaARCSolver(model_name=model_name, device=device)
    
    # 옵티마이저 초기화
    optimizer = torch.optim.Adam(
        [p for n, p in model.named_parameters() if 'encoder_projector' in n],
        lr=1e-4
    )
    
    # 솔루션 트래커 초기화
    tracker = SolutionTracker(task)
    
    # 훈련 루프
    for step in tqdm(range(num_steps), desc=f"Training task {task.task_id}"):
        # 태스크의 훈련 예제에서 입력/출력 쌍 가져오기
        for input_grid, target_grid in task.examples[:-1]:  # 마지막 예제는 테스트용
            # 훈련 단계 수행
            solution, metrics = model.train_step(
                input_grid, 
                target_grid, 
                examples=[ex for ex in task.examples[:-1] if ex[0] != input_grid],
                optimizer=optimizer
            )
            
            # 솔루션 및 메트릭 로깅
            tracker.log_solution(step, solution, metrics)
        
        # 일정 간격으로 시각화
        if (step + 1) % 50 == 0:
            visualize_solution(tracker, f"{output_dir}/step_{step+1}")
    
    # 최적 솔루션 선택
    best_solution, best_step = tracker.get_best_solution()
    
    # 최종 테스트 입력에 대한 솔루션 생성
    final_solution, _ = model.solve(task.test_input, examples=task.examples[:-1])
    
    # 결과 저장
    import json
    result = {
        "best_solution": best_solution if isinstance(best_solution, list) else best_solution.tolist(),
        "best_step": best_step,
        "final_solution": final_solution if isinstance(final_solution, list) else final_solution.tolist(),
        "metrics": tracker.metrics_history.get(best_step, {})
    }
    
    with open(os.path.join(output_dir, "best_solution.json"), 'w') as f:
        json.dump(result, f, indent=2)
    
    return final_solution

def worker_function(task_queue, result_dict, gpu_id, model_name, output_base_dir, num_steps):
    """
    병렬 처리를 위한 작업자 함수
    
    Args:
        task_queue: 처리할 태스크 큐
        result_dict: 결과를 저장할 공유 딕셔너리
        gpu_id: 사용할 GPU ID
        model_name: 사용할 모델 이름
        output_base_dir: 결과 저장 기본 경로
        num_steps: 훈련 스텝 수
    """
    # GPU 설정
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    
    while not task_queue.empty():
        try:
            task = task_queue.get(block=False)
            task_id = task.task_id
            
            print(f"GPU {gpu_id} - Processing task {task_id}")
            output_dir = os.path.join(output_base_dir, task_id)
            
            # 태스크 처리
            solution = run_single_task(
                task, 
                output_dir, 
                model_name=model_name,
                num_steps=num_steps,
                device=device
            )
            
            # 결과 저장
            if solution is not None:
                result_dict[task_id] = solution
                
        except multiprocessing.queues.Empty:
            break
        except Exception as e:
            print(f"GPU {gpu_id} - Error processing task {task_id}: {e}")

def main():
    """모든 태스크 처리를 위한 메인 함수"""
    parser = argparse.ArgumentParser(description='Run ARC-AGI solver on all tasks')
    parser.add_argument('--split', default='training', help='Data split (training/evaluation/test)')
    parser.add_argument('--output-dir', default='results/', help='Output directory')
    parser.add_argument('--model-name', default="barc0/Llama-3.1-ARC-Potpourri-Transduction-8B", help='Model name')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel')
    parser.add_argument('--gpu-ids', default='0', help='Comma-separated GPU IDs')
    parser.add_argument('--num-steps', type=int, default=1500, help='Number of training steps')
    args = parser.parse_args()
    
    # 태스크 로드
    tasks = load_tasks(args.split)
    
    # 출력 디렉토리 생성
    output_base_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(output_base_dir, exist_ok=True)
    
    if args.parallel:
        # GPU ID 파싱
        gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]
        
        # 태스크 큐 설정
        task_queue = multiprocessing.Queue()
        for task in tasks:
            task_queue.put(task)
        
        # 결과를 저장할 공유 매니저 딕셔너리
        manager = multiprocessing.Manager()
        result_dict = manager.dict()
        
        # 프로세스 시작
        processes = []
        for gpu_id in gpu_ids:
            p = multiprocessing.Process(
                target=worker_function, 
                args=(task_queue, result_dict, gpu_id, args.model_name, output_base_dir, args.num_steps)
            )
            p.start()
            processes.append(p)
        
        # 모든 프로세스 완료 대기
        for p in processes:
            p.join()
        
        # 결과 딕셔너리를 일반 딕셔너리로 변환
        results = dict(result_dict)
        
    else:
        # 순차 처리
        results = {}
        for task in tqdm(tasks, desc="Processing tasks"):
            output_dir = os.path.join(output_base_dir, task.task_id)
            solution = run_single_task(
                task, 
                output_dir, 
                model_name=args.model_name,
                num_steps=args.num_steps
            )
            if solution is not None:
                results[task.task_id] = solution
    
    # 제출 파일 생성 (test split인 경우)
    if args.split == 'test' and results:
        submission_path = os.path.join(args.output_dir, "submission.json")
        generate_submission_file(results, submission_path)
        print(f"Submission file created at {submission_path}")

if __name__ == "__main__":
    main()
