import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.random_network import RandomNetwork
from utils.tokenization import tokenize_grid, decode_grid_from_tokens
from utils.grid_preprocessing import format_grid, format_examples

class LlamaARCSolver:
    """
    Llama 기반 ARC 문제 해결 모델 - latent 벡터를 직접 최적화하는 방식
    """
    def __init__(self, model_name="barc0/Llama-3.1-ARC-Potpourri-Transduction-8B", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model {model_name} on {self.device}...")
        
        # Llama 모델 초기화
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 모델의 hidden size 가져오기
        self.hidden_size = self.model.config.hidden_size
        
        # Random network 초기화
        self.random_network = RandomNetwork(hidden_size=self.hidden_size).to(self.device)
        
        # task 별 latent 벡터는 externally 관리 (optimize_latent 메소드에서 처리)
        self.latent = None
    
    def init_latent(self, std_dev=0.01):
        """
        latent 벡터 초기화 (task별로 호출)
        
        Args:
            std_dev: 초기화를 위한 표준편차
        """
        # 랜덤 초기화된 latent 벡터 생성 (requires_grad=True로 설정하여 최적화 대상으로 지정)
        latent = torch.randn(1, self.hidden_size, device=self.device) * std_dev
        latent = nn.Parameter(latent, requires_grad=True)
        
        return latent
    
    def calculate_kl_divergence(self, latent):
        """
        Random network와의 KL divergence 계산
        
        Args:
            latent: 최적화 중인 latent 벡터
        """
        # Random network 출력
        random_output = self.random_network(latent)
        
        # KL divergence 계산
        kl_div = F.kl_div(
            F.log_softmax(latent, dim=-1),
            F.softmax(random_output, dim=-1),
            reduction='batchmean'
        )
        
        return kl_div
    
    def optimize_latent(self, input_grid, target_grid, examples=None, latent=None, optimizer=None, num_iterations=100):
        """
        latent 벡터 최적화를 통한 ARC 문제 해결
        
        Args:
            input_grid: 입력 그리드
            target_grid: 목표 출력 그리드
            examples: 추가 예제 쌍
            latent: 기존 latent 벡터 (None이면 새로 초기화)
            optimizer: 옵티마이저 (None이면 새로 생성)
            num_iterations: 최적화 반복 횟수
        """
        # latent 벡터 초기화 (없는 경우)
        if latent is None:
            latent = self.init_latent()
        
        # 옵티마이저 초기화 (없는 경우)
        if optimizer is None:
            optimizer = torch.optim.Adam([latent], lr=0.01)
        
        # 입력 처리
        example_text = format_examples(examples) if examples else ""
        input_text = example_text + "Input:\n" + format_grid(input_grid) + "\nOutput:\n"
        target_text = format_grid(target_grid)
        
        # 입력 및 타겟 토큰화
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        targets = self.tokenizer(target_text, return_tensors="pt").to(self.device)
        
        # 최적화 루프
        losses = []
        best_loss = float('inf')
        best_solution = None
        
        for i in range(num_iterations):
            # KL divergence 계산
            kl_div = self.calculate_kl_divergence(latent)
            
            # latent 벡터를 사용하여 생성
            # cross-attention 또는 input injection 방식으로 latent를 모델에 전달
            # 여기서는 간단히 입력 임베딩에 latent를 더하는 방식으로 구현
            
            # 입력 임베딩 계산
            inputs_embeds = self.model.get_input_embeddings()(inputs.input_ids)
            
            # latent를 임베딩에 주입 (첫 토큰에만 적용)
            inputs_embeds[:, 0, :] += latent
            
            # 모델 출력
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs.attention_mask,
                labels=targets.input_ids
            )
            
            # 손실 계산
            reconstruction_loss = outputs.loss
            beta = 0.1  # KL divergence 가중치
            total_loss = reconstruction_loss + beta * kl_div
            
            # 역전파 및 옵티마이저 스텝
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 손실 기록
            loss_value = total_loss.item()
            losses.append(loss_value)
            
            # 현재 최선의 솔루션 저장
            if reconstruction_loss.item() < best_loss:
                best_loss = reconstruction_loss.item()
                
                # 현재 latent로 솔루션 생성
                with torch.no_grad():
                    inputs_embeds = self.model.get_input_embeddings()(inputs.input_ids)
                    inputs_embeds[:, 0, :] += latent
                    
                    gen_outputs = self.model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=512,
                        temperature=0.0  # 결정적 출력
                    )
                
                best_solution = decode_grid_from_tokens(
                    gen_outputs[0][inputs.input_ids.shape[1]:], 
                    self.tokenizer
                )
        
        # 최종 메트릭
        metrics = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_divergence': kl_div.item(),
            'loss_history': losses
        }
        
        return best_solution, metrics, latent
    
    def solve(self, input_grid, examples=None, latent=None, num_iterations=100):
        """
        ARC 문제 해결 (latent 최적화 방식)
        
        Args:
            input_grid: 해결할 입력 그리드
            examples: 입출력 예제 쌍 (옵션)
            latent: 사전 최적화된 latent 벡터 (None이면 새로 초기화)
            num_iterations: 최적화 반복 횟수
        """
        # latent가 없으면 초기화
        if latent is None:
            latent = self.init_latent()
        
        # 예제로부터 latent 최적화
        if examples:
            optimizer = torch.optim.Adam([latent], lr=0.01)
            for input_ex, target_ex in examples:
                _, _, latent = self.optimize_latent(
                    input_ex, target_ex, 
                    examples=examples, 
                    latent=latent, 
                    optimizer=optimizer,
                    num_iterations=num_iterations
                )
        
        # 최적화된 latent로 테스트 입력에 대한 출력 생성
        example_text = format_examples(examples) if examples else ""
        input_text = example_text + "Input:\n" + format_grid(input_grid) + "\nOutput:\n"
        
        # 토큰화
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # KL divergence 계산
        kl_div = self.calculate_kl_divergence(latent)
        
        # 입력 임베딩에 latent 주입
        with torch.no_grad():
            inputs_embeds = self.model.get_input_embeddings()(inputs.input_ids)
            inputs_embeds[:, 0, :] += latent
            
            # 출력 생성
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                temperature=0.7
            )
        
        # 출력을 그리드로 디코딩
        output_grid = decode_grid_from_tokens(
            outputs[0][inputs.input_ids.shape[1]:], 
            self.tokenizer
        )
        
        return output_grid, kl_div.item(), latent
