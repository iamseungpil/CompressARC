import argparse
import os

from utils.grid_preprocessing import load_tasks
from models.llama_arc_solver import LlamaARCSolver
from evaluation.submission_generator import generate_submission_file, collect_results_from_directory

def generate_submission_from_model(model_path=None, output_path="submission.json"):
    """
    모델을 사용하여 테스트 분할의 모든 태스크에 대한 솔루션 생성 및 제출 파일 저장
    
    Args:
        model_path: 사용할 모델 경로 (None이면 기본 모델 사용)
        output_path: 제출 파일 저장 경로
    """
    # 테스트 태스크 로드
    tasks = load_tasks("test")
    
    # 모델 로드
    model = LlamaARCSolver(model_path=model_path)
    
    # 각 태스크에 대한 솔루션 생성
    solutions_dict = {}
    for task in tasks:
        print(f"Solving task {task.task_id}...")
        solution, _ = model.solve(task.test_input, examples=task.examples[:-1])
        solutions_dict[task.task_id] = solution
    
    # 제출 파일 생성
    generate_submission_file(solutions_dict, output_path)

def generate_submission_from_directory(results_dir, output_path="submission.json"):
    """
    기존 결과 디렉토리에서 솔루션 수집 및 제출 파일 생성
    
    Args:
        results_dir: 결과가 저장된 디렉토리
        output_path: 제출 파일 저장 경로
    """
    # 결과 디렉토리에서 솔루션 수집
    solutions_dict = collect_results_from_directory(results_dir)
    
    # 제출 파일 생성
    generate_submission_file(solutions_dict, output_path)

def main():
    parser = argparse.ArgumentParser(description="Generate ARC-AGI submission file")
    parser.add_argument("--output", default="submission.json", help="Output file path")
    parser.add_argument("--model-path", default=None, help="Path to fine-tuned model")
    parser.add_argument("--results-dir", default=None, help="Directory with model results")
    args = parser.parse_args()
    
    if args.results_dir:
        # 결과 디렉토리에서 생성
        generate_submission_from_directory(args.results_dir, args.output)
    else:
        # 모델에서 직접 생성
        generate_submission_from_model(args.model_path, args.output)

if __name__ == "__main__":
    main()
