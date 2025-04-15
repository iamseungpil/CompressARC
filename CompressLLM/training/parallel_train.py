import multiprocessing
import torch
import os
import argparse
from tqdm import tqdm
import json

from training.llama_train import train_model

def worker_function(task_queue, gpu_id, model_name, output_base_dir):
    """
    병렬 처리를 위한 작업자 함수
    
    Args:
        task_queue: 처리할 태스크 큐
        gpu_id: 사용할 GPU ID
        model_name: 사용할 모델 이름
        output_base_dir: 결과 저장 기본 경로
    """
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    while not task_queue.empty():
        try:
            task = task_queue.get(block=False)
            task_id = task.task_id
            
            print(f"GPU {gpu_id} - Processing task {task_id}")
            output_dir = os.path.join(output_base_dir, task_id)
            
            # 이미 처리된 태스크인지 확인
            if os.path.exists(os.path.join(output_dir, "best_solution.json")):
                print(f"Task {task_id} already processed. Skipping.")
                continue
            
            # 태스크 훈련
            try:
                train_model(task, output_dir, model_name=model_name)
                print(f"GPU {gpu_id} - Completed task {task_id}")
            except Exception as e:
                print(f"GPU {gpu_id} - Error processing task {task_id}: {e}")
                
        except multiprocessing.queues.Empty:
            break

def parallel_train(tasks, gpu_ids, model_name, output_base_dir):
    """
    여러 GPU에서 병렬로 태스크 훈련
    
    Args:
        tasks: 훈련할 태스크 목록
        gpu_ids: 사용할 GPU ID 목록
        model_name: 사용할 모델 이름
        output_base_dir: 결과 저장 기본 경로
    """
    # 태스크 큐 설정
    task_queue = multiprocessing.Queue()
    for task in tasks:
        task_queue.put(task)
    
    # 프로세스 시작
    processes = []
    for gpu_id in gpu_ids:
        p = multiprocessing.Process(
            target=worker_function, 
            args=(task_queue, gpu_id, model_name, output_base_dir)
        )
        p.start()
        processes.append(p)
    
    # 모든 프로세스 완료 대기
    for p in processes:
        p.join()

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Parallel training for ARC tasks")
    parser.add_argument("--split", type=str, default="training", help="Data split (training/evaluation/test)")
    parser.add_argument("--gpu-ids", type=str, default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--model-name", type=str, default="barc0/Llama-3.1-ARC-Potpourri-Transduction-8B", help="Model name")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    # 태스크 로드
    from utils.grid_preprocessing import load_tasks
    tasks = load_tasks(args.split)
    
    # GPU ID 파싱
    gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]
    
    # 병렬 훈련 실행
    output_base_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(output_base_dir, exist_ok=True)
    parallel_train(tasks, gpu_ids, args.model_name, output_base_dir)

if __name__ == "__main__":
    main()
