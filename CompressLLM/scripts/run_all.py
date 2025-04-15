import os
import argparse
import multiprocessing
from tqdm import tqdm
import torch
import json
import numpy as np

from models.llama_arc_solver import LlamaARCSolver
from training.solution_tracker import SolutionTracker
from utils.grid_preprocessing import load_tasks
from evaluation.submission_generator import generate_submission_file
from utils.visualization import visualize_solution
from training.llama_train import save_multi_latents

def run_single_task_multi_latent(task, output_dir, model_name=None, num_steps=30, iterations_per_step=50, device=None):
    """다중 텐서 잠재 벡터를 사용한 단일 태스크 처리"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미 처리된 태스크인지 확인
    if os.path.exists(os.path.join(output_dir, "best_solution.json")):
        print(f"Task {task.task_id} already processed. Loading previous results.")
        
        # 이전 결과 로드
        with open(os.path.join(output_dir, "best_solution.json"), 'r') as f:
            result = json.load(f)
            best_solution = result["solution"]
        
        # 테스트 솔루션 로드 (없을 경우 None 반환)
        test_solution_path = os.path.join(output_dir, "test_solution.json")
        if os.path.exists(test_solution_path):
            with open(test_solution_path, 'r') as f:
                test_result = json.load(f)
                test_solution = test_result["solution"]
            return test_solution
        
        return best_solution
    
    # 모델 초기화
    model = LlamaARCSolver(model_name=model_name, device=device)
    
    # 솔루션 트래커 초기화
    tracker = SolutionTracker(task)
    
    # 훈련 루프
    for step in tqdm(range(num_steps), desc=f"Training task {task.task_id}"):
        # 태스크의 훈련 예제에서 입력/출력 쌍 가져오기
        for input_grid, target_grid in task.examples[:-1]:  # 마지막 예제는 테스트용
            # 다중 텐서 잠재 벡터 최적화
            solution, metrics, processed_latents = model.optimize_multi_latent(
                input_grid, 
                target_grid, 
                examples=[ex for ex in task.examples[:-1] if ex[0] != input_grid],
                num_iterations=iterations_per_step
            )
            
            # 솔루션 및 메트릭 로깅 (메인 잠재 벡터를 SolutionTracker에 저장)
            main_dims = max(processed_latents.keys(), key=sum)
            main_latent = processed_latents[main_dims]
            tracker.log_solution(step, solution, metrics, main_latent)
            
            # 모든 잠재 벡터를 별도로 저장 (주요 체크포인트에서만)
            if (step + 1) % 10 == 0:
                save_multi_latents(processed_latents, os.path.join(output_dir, f"step_{step+1}_latents.npz"))
        
        # 일정 간격으로 시각화
        if (step + 1) % 10 == 0:
            visualize_solution(tracker, f"{output_dir}/step_{step+1}")
    
    # 최적 솔루션 선택
    best_solution, best_step, best_latent = tracker.get_best_solution()
    
    # 결과 저장
    result = {
        "best_solution": best_solution if isinstance(best_solution, list) else best_solution.tolist(),
        "best_step": best_step,
        "metrics": tracker.metrics_history.get(best_step, {})
    }
    
    # 메인 latent 벡터 저장
    if best_latent is not None:
        if isinstance(best_latent, torch.nn.Parameter) or isinstance(best_latent, torch.Tensor):
            result["latent"] = best_latent.detach().cpu().numpy().tolist()
    
    with open(os.path.join(output_dir, "best_solution.json"), 'w') as f:
        json.dump(result, f, indent=2)
    
    # 테스트 입력에 대한 솔루션 생성 (다중 텐서 방식 사용)
    test_solution, _, processed_latents = model.solve_with_multi_latent(
        task.test_input, 
        examples=task.examples[:-1],
        num_iterations=100  # 테스트 시 더 많은 반복 수행
    )
    
    # 최종 다중 텐서 잠재 벡터 저장
    save_multi_latents(processed_latents, os.path.join(output_dir, "final_latents.npz"))
    
    # 테스트 솔루션 저장
    test_result = {
        "solution": test_solution if isinstance(test_solution, list) else test_solution.tolist()
    }
    with open(os.path.join(output_dir, "test_solution.json"), 'w') as f:
        json.dump(test_result, f, indent=2)
    
    return test_solution

def run_single_task(task, output_dir, model_name=None, num_steps=30, iterations_per_step=50, device=None, use_multi_latent=True):
    """
    단일 태스크에 대한 latent 최적화 및 평가 실행
    
    Args:
        use_multi_latent: 다중 텐서 잠재 벡터 모드 사용 여부
    """
    if use_multi_latent:
        return run_single_task_multi_latent(task, output_dir, model_name, num_steps, iterations_per_step, device)
    
    # 이하는 기존 단일 latent 모드 코드
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미 처리된 태스크인지 확인
    if os.path.exists(os.path.join(output_dir, "best_solution.json")):
        print(f"Task {task.task_id} already processed. Loading previous results.")
        
        # 이전 결과 로드
        with open(os.path.join(output_dir, "best_solution.json"), 'r') as f:
            result = json.load(f)
            best_solution = result["solution"]
        
        # 테스트 솔루션 로드 (없을 경우 None 반환)
        test_solution_path = os.path.join(output_dir, "test_solution.json")
        if os.path.exists(test_solution_path):
            with open(test_solution_path, 'r') as f:
                test_result = json.load(f)
                test_solution = test_result["solution"]
            return test_solution
        
        return best_solution
    
    # 모델 초기화
    model = LlamaARCSolver(model_name=model_name, device=device)
    
    # 솔루션 트래커 초기화
    tracker = SolutionTracker(task)
    
    # latent 벡터 초기화
    latent = model.init_latent()
    
    # 훈련 루프
    for step in tqdm(range(num_steps), desc=f"Training task {task.task_id}"):
        # 태스크의 훈련 예제에서 입력/출력 쌍 가져오기
        for input_grid, target_grid in task.examples[:-1]:  # 마지막 예제는 테스트용
            # latent 최적화
            solution, metrics, latent = model.optimize_latent(
                input_grid, 
                target_grid, 
                examples=[ex for ex in task.examples[:-1] if ex[0] != input_grid],
                latent=latent,
                optimizer=torch.optim.Adam([latent], lr=0.01),
                num_iterations=iterations_per_step
            )
            
            # 솔루션 및 메트릭 로깅
            tracker.log_solution(step, solution, metrics, latent)
        
        # 일정 간격으로 시각화
        if (step + 1) % 10 == 0:
            visualize_solution(tracker, f"{output_dir}/step_{step+1}")
    
    # 최적 솔루션 선택
    best_solution, best_step, best_latent = tracker.get_best_solution()
    
    # 결과 저장
    result = {
        "best_solution": best_solution if isinstance(best_solution, list) else best_solution.tolist(),
        "best_step": best_step,
        "metrics": tracker.metrics_history.get(best_step, {})
    }
    
    # latent 벡터 저장
    if best_latent is not None:
        if isinstance(best_latent, torch.nn.Parameter) or isinstance(best_latent, torch.Tensor):
            result["latent"] = best_latent.detach().cpu().numpy().tolist()
    
    with open(os.path.join(output_dir, "best_solution.json"), 'w') as f:
        json.dump(result, f, indent=2)
    
    # 최종 테스트 입력에 대한 솔루션 생성
    test_solution, _, _ = model.solve(
        task.test_input, 
        examples=task.examples[:-1],
        latent=best_latent,
        num_iterations=100  # 테스트 시 더 많은 반복 수행
    )
    
    # 테스트 솔루션 저장
    test_result = {
        "solution": test_solution if isinstance(test_solution, list) else test_solution.tolist()
    }
    with open(os.path.join(output_dir, "test_solution.json"), 'w') as f:
        json.dump(test_result, f, indent=2)
    
    return test_solution

def worker_function(task_queue, result_dict, gpu_id, model_name, output_base_dir, num_steps, iterations_per_step, use_multi_latent):
    """
    병렬 처리를 위한 작업자 함수
    
    Args:
        task_queue: 처리할 태스크 큐
        result_dict: 결과를 저장할 공유 딕셔너리
        gpu_id: 사용할 GPU ID
        model_name: 사용할 모델 이름
        output_base_dir: 결과 저장 기본 경로
        num_steps: 훈련 스텝 수
        iterations_per_step: 각 스텝에서 latent 최적화 반복 횟수
        use_multi_latent: 다중 텐서 잠재 벡터 모드 사용 여부
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
            test_solution = run_single_task(
                task, 
                output_dir, 
                model_name=model_name,
                num_steps=num_steps,
                iterations_per_step=iterations_per_step,
                device=device,
                use_multi_latent=use_multi_latent
            )
            
            # 결과 저장
            if test_solution is not None:
                result_dict[task_id] = test_solution
                
        except multiprocessing.queues.Empty:
            break
        except Exception as e:
            print(f"GPU {gpu_id} - Error processing task {task_id}: {e}")

def main():
    """모든 태스크 처리를 위한 메인 함수"""
    parser = argparse.ArgumentParser(description='Run ARC-AGI solver on all tasks with latent optimization')
    parser.add_argument('--split', default='training', help='Data split (training/evaluation/test)')
    parser.add_argument('--output-dir', default='results/', help='Output directory')
    parser.add_argument('--model-name', default="barc0/Llama-3.1-ARC-Potpourri-Transduction-8B", help='Model name')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel')
    parser.add_argument('--gpu-ids', default='0', help='Comma-separated GPU IDs')
    parser.add_argument('--num-steps', type=int, default=30, help='Number of training steps')
    parser.add_argument('--iterations-per-step', type=int, default=50, help='Number of iterations per step')
    parser.add_argument('--use-multi-latent', action='store_true', help='Use multi-tensor latent mode (CompressARC style)')
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
                args=(
                    task_queue, 
                    result_dict, 
                    gpu_id, 
                    args.model_name, 
                    output_base_dir, 
                    args.num_steps,
                    args.iterations_per_step,
                    args.use_multi_latent
                )
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
            test_solution = run_single_task(
                task, 
                output_dir, 
                model_name=args.model_name,
                num_steps=args.num_steps,
                iterations_per_step=args.iterations_per_step,
                use_multi_latent=args.use_multi_latent
            )
            if test_solution is not None:
                results[task.task_id] = test_solution
    
    # 제출 파일 생성 (test split인 경우)
    if args.split == 'test' and results:
        submission_path = os.path.join(args.output_dir, "submission.json")
        generate_submission_file(results, submission_path)
        print(f"Submission file created at {submission_path}")

if __name__ == "__main__":
    main()
