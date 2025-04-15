import numpy as np
import torch

class SolutionTracker:
    """
    훈련 중 솔루션 성능을 추적하고 최적의 솔루션을 선택하는 클래스
    (기존 CompressARC의 Logger 클래스와 유사한 기능)
    """
    def __init__(self, task):
        self.task = task
        self.solutions_history = {}  # 각 스텝별 생성된 솔루션 저장
        self.metrics_history = {}    # 각 스텝별 성능 지표
        self.best_step = None
        self.best_solution = None
        
        # KL 및 재구성 오류 커브 추적 (분석용)
        self.kl_curves = {}
        self.reconstruction_error_curve = []
        
    def log_solution(self, step, solution, metrics):
        """훈련 단계에서 생성된 솔루션과 성능 지표 기록"""
        self.solutions_history[step] = solution
        self.metrics_history[step] = metrics
        
        # 메트릭 추적
        self.reconstruction_error_curve.append(metrics['reconstruction_loss'])
        
        # KL 기록 (단순화된 버전)
        if 'kl_divergence' in metrics:
            self.kl_curves.setdefault('kl_divergence', []).append(metrics['kl_divergence'])
        
        # 최적 솔루션 갱신 (재구성 손실 기준)
        if self.best_step is None or metrics['reconstruction_loss'] < self.metrics_history.get(self.best_step, {}).get('reconstruction_loss', float('inf')):
            self.best_step = step
            self.best_solution = solution
    
    def get_best_solution(self):
        """최적의 솔루션 반환"""
        return self.best_solution, self.best_step
    
    def get_metrics_history(self):
        """메트릭 히스토리 반환 (시각화용)"""
        return {
            'reconstruction_error': self.reconstruction_error_curve,
            'kl_curves': self.kl_curves
        }
