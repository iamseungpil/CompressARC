import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def visualize_solution(tracker, output_path, show_kl=True):
    """
    솔루션 및 메트릭 시각화
    
    Args:
        tracker: SolutionTracker 객체
        output_path: 출력 파일 경로 (확장자 제외)
        show_kl: KL divergence 곡선 시각화 여부
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 최적 솔루션 시각화
    best_solution, best_step = tracker.get_best_solution()
    visualize_grid(best_solution, f"{output_path}_best_solution.png", 
                   title=f"Best Solution (Step {best_step})")
    
    # 메트릭 시각화
    metrics_history = tracker.get_metrics_history()
    
    # 재구성 오류 곡선
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_history['reconstruction_error'], label='Reconstruction Error')
    plt.axvline(x=best_step, color='r', linestyle='--', label=f'Best Step ({best_step})')
    plt.xlabel('Training Step')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error over Training')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}_reconstruction_error.png", bbox_inches='tight')
    plt.close()
    
    # KL divergence 곡선 (있는 경우)
    if show_kl and 'kl_curves' in metrics_history and metrics_history['kl_curves']:
        plt.figure(figsize=(10, 6))
        for name, curve in metrics_history['kl_curves'].items():
            plt.plot(curve, label=name)
        plt.axvline(x=best_step, color='r', linestyle='--', label=f'Best Step ({best_step})')
        plt.xlabel('Training Step')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence over Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_path}_kl_divergence.png", bbox_inches='tight')
        plt.close()

def visualize_grid(grid, output_path, title=None):
    """
    ARC 그리드 시각화
    
    Args:
        grid: ARC 그리드
        output_path: 출력 파일 경로
        title: 그래프 제목 (옵션)
    """
    # 색상 설정
    color_map = plt.get_cmap('tab10')
    
    # 그리드 시각화
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 그리드에 사용된 고유 값 확인
    unique_values = np.unique(grid)
    colors = [color_map(val % 10) for val in range(len(unique_values))]
    cmap = mcolors.ListedColormap(colors)
    
    # 그리드 표시
    im = ax.imshow(grid, cmap=cmap, interpolation='nearest')
    
    # 격자 추가
    ax.set_xticks(np.arange(-0.5, len(grid[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    
    # 셀 값 표시
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            ax.text(j, i, str(grid[i][j]), ha='center', va='center', 
                   color='white' if grid[i][j] > 4 else 'black', fontsize=12)
    
    # 컬러바 추가
    cbar = plt.colorbar(im, ax=ax, ticks=range(len(unique_values)))
    cbar.set_ticklabels([str(val) for val in unique_values])
    
    # 제목 설정 (있는 경우)
    if title:
        plt.title(title)
    
    # 저장
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
