import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.random_network import RandomNetwork, MultiRandomNetwork
from models.multitensor_latent import MultiTensorLatent
from models.directional_processors import DirectionalProcessor, CumMaxLayer, ShiftLayer
from utils.tokenization import tokenize_grid, decode_grid_from_tokens
from utils.grid_preprocessing import format_grid, format_examples
from utils.multitensor_operations import multitensor_normalize, multitensor_apply, multitensor_kl_divergence, share_information, grid_to_multitensor

class LlamaARCSolver:
    """
    Llama 기반 ARC 문제 해결 모델 - CompressARC 스타일의 다중 텐서 잠재 벡터 최적화 방식
    """
    def __init__(self, model_name="barc0/Llama-3.1-ARC-Potpourri-Transduction-8B", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model {model_name} on {self.device}...")
        
        # Llama 모델 초기화
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 모델의 hidden size 가져오기
        self.hidden_size = self.model.config.hidden_size
        
        # 다중 텐서 잠재 벡터 시스템 초기화
        self.multitensor_latent = MultiTensorLatent(base_hidden_size=self.hidden_size, device=self.device)
        
        # 다중 랜덤 네트워크 초기화 (KL 발산 계산용)
        self.multi_random_network = MultiRandomNetwork(
            self.multitensor_latent.dims_list, 
            base_hidden_size=self.hidden_size
        ).to(self.device)
        
        # 방향성 처리 모듈 초기화
        self.directional_processor = DirectionalProcessor(self.hidden_size).to(self.device)
        self.cummax_layer = CumMaxLayer(self.hidden_size).to(self.device)
        self.shift_layer = ShiftLayer(self.hidden_size).to(self.device)
        
        # 텐서 간 정보 공유를 위한 가중치
        self.share_up_weights = nn.ModuleDict({
            str(tuple(dims)): nn.Sequential(
                nn.Linear(self.multitensor_latent.get_hidden_size(dims), self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.multitensor_latent.get_hidden_size(dims))
            ).to(self.device)
            for dims in self.multitensor_latent.dims_list
        })
        
        self.share_down_weights = nn.ModuleDict({
            str(tuple(dims)): nn.Sequential(
                nn.Linear(self.multitensor_latent.get_hidden_size(dims), self.hidden_size // 2),
                nn.GELU(),
                nn.Linear(self.hidden_size // 2, self.multitensor_latent.get_hidden_size(dims))
            ).to(self.device)
            for dims in self.multitensor_latent.dims_list
        })
        
        # 기존 호환을 위한 latent 변수 유지
        self.latent = None
    
    def get_parameters(self):
        """최적화를 위한 모든 파라미터 반환"""
        params = []
        
        # 다중 텐서 잠재 벡터 파라미터
        params.extend(self.multitensor_latent.get_all_parameters())
        
        # 방향성 처리 모듈 파라미터
        params.extend(list(self.directional_processor.parameters()))
        params.extend(list(self.cummax_layer.parameters()))
        params.extend(list(self.shift_layer.parameters()))
        
        # 공유 가중치 파라미터
        for module in self.share_up_weights.values():
            params.extend(list(module.parameters()))
        
        for module in self.share_down_weights.values():
            params.extend(list(module.parameters()))
        
        return params
    
    def init_latent(self, std_dev=0.01):
        """
        기존 코드와의 호환성을 위한 단일 latent 벡터 초기화 메서드
        
        Args:
            std_dev: 초기화를 위한 표준편차
        """
        # 랜덤 초기화된 latent 벡터 생성
        latent = torch.randn(1, self.hidden_size, device=self.device) * std_dev
        latent = nn.Parameter(latent, requires_grad=True)
        
        return latent
    
    def calculate_multi_kl_divergence(self):
        """
        다중 텐서 잠재 벡터와 랜덤 네트워크 간의 KL 발산 계산
        """
        # 다중 텐서 잠재 벡터 가져오기
        latent_dict = {
            tuple(dims): self.multitensor_latent.get_latent(dims)
            for dims in self.multitensor_latent.dims_list
        }
        
        # 각 잠재 벡터에 랜덤 네트워크 적용
        random_outputs = {}
        for dims in self.multitensor_latent.dims_list:
            dims_key = tuple(dims)
            random_outputs[dims_key] = self.multi_random_network.forward_for_dims(dims, latent_dict[dims_key])
        
        # KL 발산 계산
        total_kl, kl_components = multitensor_kl_divergence(latent_dict, random_outputs)
        
        return total_kl, kl_components
    
    def calculate_kl_divergence(self, latent):
        """
        기존 코드와의 호환성을 위한 단일 latent 벡터의 KL 발산 계산
        
        Args:
            latent: 최적화 중인 latent 벡터
        """
        # Random network 출력
        random_output = self.multi_random_network.forward_for_dims(
            self.multitensor_latent.dims_list[0], latent
        )
        
        # KL divergence 계산
        kl_div = F.kl_div(
            F.log_softmax(latent, dim=-1),
            F.softmax(random_output, dim=-1),
            reduction='batchmean'
        )
        
        return kl_div
    
    def process_latent_tensors(self):
        """
        다중 텐서 잠재 벡터 처리 (CompressARC 스타일)
        - 다중 텐서 간 정보 공유
        - 방향성 처리
        - 정규화
        """
        try:
            # 다중 텐서 잠재 벡터 가져오기
            latent_dict = {
                tuple(dims): self.multitensor_latent.get_latent(dims)
                for dims in self.multitensor_latent.dims_list
            }
            
            # 1. 상향 공유 (하위->상위 텐서)
            shared_up = share_information(latent_dict, direction='up')
            
            # 각 텐서에 가중치 적용
            for dims_key in shared_up:
                dims_str = str(dims_key)
                if dims_str in self.share_up_weights:
                    shared_up[dims_key] = self.share_up_weights[dims_str](shared_up[dims_key])
                else:
                    print(f"Warning: Missing up-weight for {dims_key}, using original values")
            
            # 2. 방향성 처리 (방향 차원이 있는 텐서만)
            for dims_key in shared_up:
                dims = list(dims_key)
                if len(dims) > 2 and dims[2] == 1:  # 방향 차원이 있는 경우 (인덱스 유효성 검사)
                    # 방향성 처리 적용
                    shared_up[dims_key] = F.gelu(shared_up[dims_key])
            
            # 3. 하향 공유 (상위->하위 텐서)
            shared_down = share_information(shared_up, direction='down')
            
            # 각 텐서에 가중치 적용
            for dims_key in shared_down:
                dims_str = str(dims_key)
                if dims_str in self.share_down_weights:
                    shared_down[dims_key] = self.share_down_weights[dims_str](shared_down[dims_key])
                else:
                    print(f"Warning: Missing down-weight for {dims_key}, using original values")
            
            # 4. 정규화
            normalized = multitensor_normalize(shared_down)
            
            return normalized
            
        except Exception as e:
            print(f"Error in process_latent_tensors: {e}")
            # 오류 발생 시 원본 잠재 벡터 그대로 반환
            return {
                tuple(dims): self.multitensor_latent.get_latent(dims)
                for dims in self.multitensor_latent.dims_list
            }
    
    def combine_latents(self, processed_latents):
        """
        처리된 다중 텐서 잠재 벡터를 단일 벡터로 결합
        """
        try:
            if not processed_latents:  # 빈 사전인 경우 처리
                print("Warning: Empty processed_latents dictionary. Using default latent.")
                return torch.zeros(1, self.hidden_size, device=self.device)
            
            # 주요 잠재 벡터 선택 (가장 많은 차원을 가진 것)
            try:
                main_dims = max(processed_latents.keys(), key=sum)
            except Exception as e:
                print(f"Error selecting main dimensions: {e}")
                main_dims = next(iter(processed_latents.keys()))  # 첫 번째 키 사용
                
            main_latent = processed_latents[main_dims].clone()  # 복사본 사용
            
            # 투영 레이어 생성 또는 가져오기
            if not hasattr(self, '_projection_layers'):
                self._projection_layers = {}
            
            # 다른 잠재 벡터의 정보를 결합
            for dims_key, latent in processed_latents.items():
                if dims_key != main_dims:
                    try:
                        # 투영 레이어 생성 또는 가져오기
                        dims_str = str(dims_key) + "_to_" + str(main_dims)
                        
                        if dims_str not in self._projection_layers:
                            source_size = self.multitensor_latent.get_hidden_size(list(dims_key))
                            target_size = self.multitensor_latent.get_hidden_size(list(main_dims))
                            
                            self._projection_layers[dims_str] = nn.Linear(
                                source_size, target_size
                            ).to(self.device)
                        
                        # 투영 후 결합 (가중치 적용)
                        projected = self._projection_layers[dims_str](latent)
                        main_latent = main_latent + 0.1 * projected
                    except Exception as e:
                        print(f"Error projecting dimension {dims_key}: {e}")
                        continue  # 해당 차원 건너뛰기
            
            return main_latent
            
        except Exception as e:
            print(f"Error in combine_latents: {e}")
            # 오류 발생 시 기본 잠재 벡터 반환
            return torch.zeros(1, self.hidden_size, device=self.device)
    
    def optimize_multi_latent(self, input_grid, target_grid, examples=None, num_iterations=100):
        """
        다중 텐서 잠재 벡터 최적화를 통한 ARC 문제 해결
        
        Args:
            input_grid: 입력 그리드
            target_grid: 목표 출력 그리드
            examples: 추가 예제 쌍
            num_iterations: 최적화 반복 횟수
        """
        # 옵티마이저 초기화
        optimizer = torch.optim.Adam(self.get_parameters(), lr=0.01)
        
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
        best_processed_latents = None
        
        for i in range(num_iterations):
            # KL 발산 계산
            total_kl, kl_components = self.calculate_multi_kl_divergence()
            
            # 다중 텐서 잠재 벡터 처리 (CompressARC 스타일)
            processed_latents = self.process_latent_tensors()
            
            # 처리된 잠재 벡터 결합
            combined_latent = self.combine_latents(processed_latents)
            
            # 입력 임베딩 계산
            inputs_embeds = self.model.get_input_embeddings()(inputs.input_ids)
            
            # 결합된 잠재 벡터를 임베딩에 주입 (첫 토큰에만 적용)
            inputs_embeds[:, 0, :] += combined_latent
            
            # 모델 출력
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs.attention_mask,
                labels=targets.input_ids
            )
            
            # 손실 계산
            reconstruction_loss = outputs.loss
            beta = 0.1  # KL 발산 가중치
            total_loss = reconstruction_loss + beta * total_kl
            
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
                best_processed_latents = {k: v.detach().clone() for k, v in processed_latents.items()}
                
                # 현재 잠재 벡터로 솔루션 생성
                with torch.no_grad():
                    combined_latent = self.combine_latents(processed_latents)
                    inputs_embeds = self.model.get_input_embeddings()(inputs.input_ids)
                    inputs_embeds[:, 0, :] += combined_latent
                    
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
            'kl_divergence': total_kl.item(),
            'kl_components': {str(k): v.item() for k, v in kl_components.items()},
            'loss_history': losses
        }
        
        return best_solution, metrics, best_processed_latents
    
    def optimize_latent(self, input_grid, target_grid, examples=None, latent=None, optimizer=None, num_iterations=100):
        """
        기존 호환성을 위한 단일 latent 벡터 최적화 메서드
        
        Args:
            input_grid: 입력 그리드
            target_grid: 목표 출력 그리드
            examples: 추가 예제 쌍
            latent: 기존 latent 벡터 (None이면 새로 초기화)
            optimizer: 옵티마이저 (None이면 새로 생성)
            num_iterations: 최적화 반복 횟수
        """
        # 다중 텐서 방식 사용
        solution, metrics, processed_latents = self.optimize_multi_latent(
            input_grid, target_grid, examples, num_iterations
        )
        
        # 기존 코드 호환을 위해 주요 잠재 벡터를 반환
        main_dims = max(processed_latents.keys(), key=sum)
        main_latent = processed_latents[main_dims]
        
        # 기존 단일 latent 변수 업데이트
        self.latent = main_latent
        
        return solution, metrics, main_latent
    
    def solve_with_multi_latent(self, input_grid, examples=None, num_iterations=100):
        """
        다중 텐서 잠재 벡터를 사용한 ARC 문제 해결
        
        Args:
            input_grid: 해결할 입력 그리드
            examples: 입출력 예제 쌍 (옵션)
            num_iterations: 최적화 반복 횟수
        """
        # 예제로부터 다중 텐서 잠재 벡터 최적화
        if examples:
            for input_ex, target_ex in examples:
                _, _, processed_latents = self.optimize_multi_latent(
                    input_ex, target_ex, 
                    examples=[ex for ex in examples if ex[0] != input_ex],
                    num_iterations=num_iterations
                )
        
        # 입력 처리
        example_text = format_examples(examples) if examples else ""
        input_text = example_text + "Input:\n" + format_grid(input_grid) + "\nOutput:\n"
        
        # 토큰화
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # 다중 텐서 잠재 벡터 처리
        processed_latents = self.process_latent_tensors()
        combined_latent = self.combine_latents(processed_latents)
        
        # KL 발산 계산
        total_kl, _ = self.calculate_multi_kl_divergence()
        
        # 입력 임베딩에 잠재 벡터 주입
        with torch.no_grad():
            inputs_embeds = self.model.get_input_embeddings()(inputs.input_ids)
            inputs_embeds[:, 0, :] += combined_latent
            
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
        
        return output_grid, total_kl.item(), processed_latents
    
    def solve(self, input_grid, examples=None, latent=None, num_iterations=100):
        """
        기존 호환성을 위한 ARC 문제 해결 메서드
        
        Args:
            input_grid: 해결할 입력 그리드
            examples: 입출력 예제 쌍 (옵션)
            latent: 사전 최적화된 latent 벡터 (None이면 새로 초기화)
            num_iterations: 최적화 반복 횟수
        """
        # 다중 텐서 방식으로 해결
        output_grid, kl_div, processed_latents = self.solve_with_multi_latent(
            input_grid, examples, num_iterations
        )
        
        # 주요 잠재 벡터 선택
        main_dims = max(processed_latents.keys(), key=sum)
        main_latent = processed_latents[main_dims]
        
        # 기존 단일 latent 변수 업데이트
        self.latent = main_latent
        
        return output_grid, kl_div, main_latent
