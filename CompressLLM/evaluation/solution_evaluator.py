import numpy as np

def evaluate_solution(output_grid, target_grid):
    """
    생성된 솔루션 평가
    
    Args:
        output_grid: 모델이 생성한 그리드
        target_grid: 실제 목표 그리드
    """
    # 그리드가 완전히 일치하는지 확인
    exact_match = np.array_equal(output_grid, target_grid)
    
    # 개별 셀 정확도 계산
    cell_accuracy = np.mean(np.array(output_grid) == np.array(target_grid))
    
    return {
        'exact_match': exact_match,
        'cell_accuracy': float(cell_accuracy)
    }

def evaluate_solutions_batch(solution_dict, groundtruth_dict):
    """
    여러 솔루션을 일괄 평가
    
    Args:
        solution_dict: {task_id: solution_grid} 형식의 딕셔너리
        groundtruth_dict: {task_id: target_grid} 형식의 딕셔너리
    """
    results = {}
    exact_matches = []
    cell_accuracies = []
    
    for task_id, solution in solution_dict.items():
        if task_id in groundtruth_dict:
            target = groundtruth_dict[task_id]
            evaluation = evaluate_solution(solution, target)
            
            results[task_id] = evaluation
            exact_matches.append(evaluation['exact_match'])
            cell_accuracies.append(evaluation['cell_accuracy'])
    
    # 전체 메트릭 계산
    overall_metrics = {
        'exact_match_rate': np.mean(exact_matches),
        'average_cell_accuracy': np.mean(cell_accuracies),
        'evaluated_tasks': len(results)
    }
    
    return results, overall_metrics

def calculate_pass_at_n(evaluations, n=1):
    """
    Pass@N 메트릭 계산 (정확히 n개 태스크를 맞출 확률)
    
    Args:
        evaluations: {task_id: {'exact_match': bool, ...}} 형식의 딕셔너리
        n: 정확히 맞춰야 할 태스크 수
    """
    exact_matches = [eval_result['exact_match'] for eval_result in evaluations.values()]
    pass_at_n = sum(exact_matches) >= n
    
    return pass_at_n
