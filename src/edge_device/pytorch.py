import os
import time
import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils.config import Config
from tqdm import trange

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Constants
COORD_SCALE = 200
COORD_OFFSET = 100
DEFAULT_Z = 70


class PyTorchEngine:
    """PyTorch 추론 엔진"""
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 디바이스 정보 로깅
        logger.info(f"Using device: {self.device}")
        if self.device == 'cuda':
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # PyTorch 모델 로드
        from model import Net
        
        # 모델 구조 생성 (입력 크기 12, 출력 2)
        self.model = Net(12, 1024, 4, output_N=2).to(self.device)
        
        # 가중치 로드
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def _warmup(self, data: np.ndarray) -> None:
        """워밍업"""
        with torch.no_grad():
            sample = torch.tensor(data[0:1], dtype=torch.float32, device=self.device)
            _ = self.model(sample)
            
            # GPU 사용 확인
            if self.device == 'cuda':
                logger.info(f"GPU Memory after warmup: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    def _prep_inp(self, data: np.ndarray) -> torch.Tensor:
        """입력 전처리"""
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def _proc_out(self, raw: torch.Tensor) -> np.ndarray:
        """출력 후처리"""
        return raw.cpu().numpy()
    
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """단일 추론"""
        n_samples = data.shape[0]
        
        # 워밍업
        self._warmup(data)
        
        results = []
        start = time.perf_counter()
        
        with torch.no_grad():
            for i in trange(n_samples, desc="PyTorch Inference", leave=False):
                batch = self._prep_inp(data[i:i+1])
                output = self.model(batch)
                results.append(self._proc_out(output))
        
        elapsed = time.perf_counter() - start
        
        # 결과를 numpy 배열로 변환
        results = np.vstack(results)
        
        return results, elapsed
    
    def predict_batch(self, data: np.ndarray, batch_size: int = 32) -> Tuple[np.ndarray, float]:
        """배치별 추론"""
        n_samples = data.shape[0]
        
        # 워밍업
        self._warmup(data)
        
        results = []
        start = time.perf_counter()
        
        with torch.no_grad():
            for i in trange(0, n_samples, batch_size, desc="PyTorch Batch Inference", leave=False):
                end_idx = min(i + batch_size, n_samples)
                batch = self._prep_inp(data[i:end_idx])
                output = self.model(batch)
                results.append(self._proc_out(output))
        
        elapsed = time.perf_counter() - start
        
        # 결과를 numpy 배열로 변환
        results = np.vstack(results)
        
        return results, elapsed


def load_data(csv_path: str, cfg: Config) -> Tuple[pd.DataFrame, np.ndarray]:
    """데이터 로딩"""
    df = pd.read_csv(csv_path, header=None)
    df = df.drop(columns=df.columns[2::3])
    
    x = torch.tensor(df.values, dtype=torch.float32, device=cfg.device)
    scaled = cfg.scaler.transform(x.cpu())
    data = np.ascontiguousarray(scaled, dtype=np.float32)
    
    return df, data


def make_results(orig_df: pd.DataFrame, preds: np.ndarray, elapsed: float) -> pd.DataFrame:
    """결과 생성"""
    adj_preds = preds * COORD_SCALE - COORD_OFFSET
    
    df = orig_df.copy()
    df[['pred_x', 'pred_y']] = adj_preds
    df['pred_z'] = DEFAULT_Z
    df['inference_time_ms'] = (elapsed / len(preds)) * 1000
    
    return df


def print_stats(n_samples: int, elapsed: float, path: str) -> None:
    """통계 출력"""
    total_ms = elapsed * 1000
    avg_ms = total_ms / n_samples
    tput = n_samples / elapsed
    
    logger.info(f"Samples: {n_samples}")
    logger.info(f"Total time: {total_ms:.3f} ms")
    logger.info(f"Avg per sample: {avg_ms:.6f} ms")
    logger.info(f"Throughput: {tput:.2f} samples/s")
    logger.info(f"Results saved to {path}")


def run_pytorch_inference(model_path: str, csv_path: str, 
                         output_path: Optional[str] = None,
                         use_batch: bool = True, 
                         batch_size: int = 32) -> Tuple[pd.DataFrame, float]:
    """PyTorch 추론 실행"""
    cfg = Config(num_users=6)
    orig_df, data = load_data(csv_path, cfg)
    
    engine = PyTorchEngine(model_path, device=cfg.device)
    
    if use_batch:
        preds, elapsed = engine.predict_batch(data, batch_size)
    else:
        preds, elapsed = engine.predict(data)
    
    results_df = make_results(orig_df, preds, elapsed)
    
    if output_path is None:
        output_path = os.path.join('src', 'edge_device', 'results_pytorch.csv')
    
    results_df.to_csv(output_path, index=False)
    print_stats(len(data), elapsed, output_path)
    
    return results_df, elapsed





if __name__ == "__main__":
    # PyTorch 모델 사용
    model_path = os.path.join('src', 'train_model', 'result', 'models', 'num_gu', 'best_num_gu_6.pt')
    csv_path = os.path.join('src', 'train_model', 'result', 'data', 'gn_coords_6.csv')
    
    # 단일 샘플 추론 (다른 엔진과 공정한 비교)
    df_results, elapsed = run_pytorch_inference(
        model_path, 
        csv_path, 
        use_batch=False, 
        batch_size=1
    )
