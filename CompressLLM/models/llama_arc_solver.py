import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.random_network import RandomNetwork
from utils.tokenization import tokenize_grid, decode_grid_from_tokens
from utils.grid_preprocessing import format_grid, format_examples

class LlamaARCSolver:
    """
    Llama 기반 ARC 문제 해결 모델
    """
    def __init__(self, model_name="barc0/Llama-3.1-ARC-Potpourri-Transduction-8B", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model {model_name} on {self.device}...")
        
        # Llama 모델 초기화
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 인코딩을 위한 파라미터 (파인튜닝 가능)
        self.encoder_projector = nn.Linear(768, 768).to(self.device)  # 임베딩 차원에 맞게 조정 필요
        
        # Random network 초기화
        self.random_network = RandomNetwork(hidden_size=768).to(self.device)
    
    def encode_grid(self, grid):
        """ARC grid를 인코딩하여 벡터 표현으로 변환"""
        # Grid를 토큰화
        tokenized_grid = tokenize_grid(grid, self.tokenizer).to(self.device)
        
        # Llama의 임베딩 추출
        with torch.no_grad():  # 기본 임베딩은 고정
            embeddings = self.model.get_input_embeddings()(tokenized_grid)
        
        # 임베딩을 인코더로 처리 (파인튜닝 가능 부분)
        encoded = self.encoder_projector(embeddings)
        
        return encoded
    
    def calculate_kl_divergence(self, encoded_representation):
        """Random network와의 KL divergence 계산"""
        # Random network 출력
        random_output = self.random_network(encoded_representation)
        
        # KL divergence 계산
        kl_div = F.kl_div(
            F.log_softmax(encoded_representation, dim=-1),
            F.softmax(random_output, dim=-1),
            reduction='batchmean'
        )
        
        return kl_div
    
    def solve(self, input_grid, examples=None):
        """
        ARC 문제 해결
        
        Args:
            input_grid: 해결할 입력 그리드
            examples: 입출력 예제 쌍 (옵션)
        """
        # 입력 그리드 토큰화
        example_text = format_examples(examples) if examples else ""
        input_text = example_text + "Input:\n" + format_grid(input_grid) + "\nOutput:\n"
        
        # 토큰화
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # 입력 인코딩
        encoded = self.encode_grid(input_grid)
        
        # KL divergence 계산
        kl_div = self.calculate_kl_divergence(encoded)
        
        # Llama 모델로 생성
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.7
        )
        
        # 출력을 그리드로 디코딩
        output_grid = decode_grid_from_tokens(
            outputs[0][inputs.input_ids.shape[1]:], 
            self.tokenizer
        )
        
        return output_grid, kl_div
    
    def train_step(self, input_grid, target_grid, examples=None, optimizer=None):
        """
        훈련 단계 수행
        
        Args:
            input_grid: 입력 그리드
            target_grid: 목표 출력 그리드
            examples: 추가 예제 쌍
            optimizer: 옵티마이저
        """
        if optimizer is None:
            raise ValueError("Optimizer must be provided for training")
        
        # 입력 처리
        example_text = format_examples(examples) if examples else ""
        input_text = example_text + "Input:\n" + format_grid(input_grid) + "\nOutput:\n"
        target_text = format_grid(target_grid)
        
        # 입력 및 타겟 토큰화
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        targets = self.tokenizer(target_text, return_tensors="pt").to(self.device)
        
        # 입력 인코딩
        encoded = self.encode_grid(input_grid)
        
        # KL divergence 계산
        kl_div = self.calculate_kl_divergence(encoded)
        
        # 모델 출력
        outputs = self.model(input_ids=inputs.input_ids, labels=targets.input_ids)
        
        # 손실 계산
        reconstruction_loss = outputs.loss
        total_loss = reconstruction_loss + 0.1 * kl_div  # beta 가중치 조정 가능
        
        # 역전파 및 옵티마이저 스텝
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 현재 솔루션 생성 (평가용)
        with torch.no_grad():
            gen_outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=512,
                temperature=0.0  # 평가시 결정적 출력
            )
        
        solution_grid = decode_grid_from_tokens(
            gen_outputs[0][inputs.input_ids.shape[1]:], 
            self.tokenizer
        )
        
        metrics = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_divergence': kl_div.item()
        }
        
        return solution_grid, metrics
