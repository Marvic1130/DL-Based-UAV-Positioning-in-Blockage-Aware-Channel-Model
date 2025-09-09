import os
import time
import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from utils.config import Config
from tqdm import trange

from utils.tools import createDirectory

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Constants
COORD_SCALE = 200
COORD_OFFSET = 100
DEFAULT_Z = 70


class ModelInfo:
    """모델 정보"""
    def __init__(self, inp_det: dict, out_det: dict):
        self.inp_dtype = inp_det['dtype']
        self.out_dtype = out_det['dtype']
        self.inp_idx = inp_det["index"]
        self.out_idx = out_det["index"]
        
        self.inp_quant = (self.inp_dtype == np.int8)
        self.out_quant = (self.out_dtype == np.int8)
        
        # 양자화 파라미터는 INT8일 때만 추출
        if self.inp_quant:
            self.inp_scale = self._get_quant_param(inp_det, 'scales', 1.0)
            self.inp_zp = self._get_quant_param(inp_det, 'zero_points', 0)
        else:
            self.inp_scale = 1.0
            self.inp_zp = 0
            
        if self.out_quant:
            self.out_scale = self._get_quant_param(out_det, 'scales', 1.0)
            self.out_zp = self._get_quant_param(out_det, 'zero_points', 0)
        else:
            self.out_scale = 1.0
            self.out_zp = 0
    
    @staticmethod
    def _get_quant_param(det: dict, param: str, default):
        """양자화 파라미터 추출"""
        quant_params = det.get('quantization_parameters', {})
        values = quant_params.get(param, [])
        return values[0] if len(values) > 0 else default


class TFLiteEngine:
    """TFLite 추론 엔진"""
    
    def __init__(self, model_path: str):
        self.interp = tf.lite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        
        inp_det = self.interp.get_input_details()[0]
        out_det = self.interp.get_output_details()[0]
        self.info = ModelInfo(inp_det, out_det)
    
    def _warmup(self, data: np.ndarray) -> None:
        """워밍업"""
        inp = self._prep_inp(data[0:1])
        self.interp.set_tensor(self.info.inp_idx, inp)
        self.interp.invoke()
    
    def _prep_inp(self, data: np.ndarray) -> np.ndarray:
        """입력 전처리"""
        if self.info.inp_quant:
            quant = ((data / self.info.inp_scale) + self.info.inp_zp).astype(np.int8)
            return quant.reshape(1, -1) if len(quant.shape) == 1 else quant
        else:
            return data.astype(self.info.inp_dtype)
    
    def _proc_out(self, raw: np.ndarray) -> np.ndarray:
        """출력 후처리"""
        if self.info.out_quant:
            return ((raw.astype(np.float32) - self.info.out_zp) * self.info.out_scale)
        return raw.flatten()
    
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """배치 추론"""
        n_samples = data.shape[0]
        
        # 워밍업
        self._warmup(data)
        
        # 결과 배열 할당
        self.interp.set_tensor(self.info.inp_idx, self._prep_inp(data[0:1]))
        self.interp.invoke()
        out_shape = self.interp.get_tensor(self.info.out_idx).shape
        results = np.empty((n_samples, out_shape[1]), dtype=np.float32)
        
        # 추론 루프
        model_type = "INT8" if self.info.inp_quant else "FP32/FP16"
        start = time.perf_counter()
        
        for i in trange(n_samples, desc=f"TFLite {model_type}", leave=False):
            inp = self._prep_inp(data[i:i+1])
            self.interp.set_tensor(self.info.inp_idx, inp)
            self.interp.invoke()
            
            raw = self.interp.get_tensor(self.info.out_idx)
            results[i] = self._proc_out(raw)
        
        elapsed = time.perf_counter() - start
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


def run_tflite_inference(model_path: str, csv_path: str, 
                        output_path: Optional[str] = None) -> Tuple[pd.DataFrame, float]:
    """TFLite 추론 실행"""
    cfg = Config(num_users=6)
    orig_df, data = load_data(csv_path, cfg)
    
    engine = TFLiteEngine(model_path)
    preds, elapsed = engine.predict(data)
    
    results_df = make_results(orig_df, preds, elapsed)
    
    if output_path is None:
        output_path = os.path.join('src', 'edge_device', 'results_tflite.csv')

    results_df.to_csv(output_path, index=False)
    print_stats(len(data), elapsed, output_path)
    
    return results_df, elapsed

if __name__ == "__main__":
    # 기본 실행

    csv_path = os.path.join('src', 'train_model', 'result', 'data', 'gn_coords_6.csv')

    model_path = os.path.join("src", "edge_device", "models", "tf_model", "model_float32.tflite")
    fp32_result_path = os.path.join('src', 'edge_device', 'results','orin_fp32.csv')
    df_results, elapsed = run_tflite_inference(model_path, csv_path, fp32_result_path)

    model_path = os.path.join("src", "edge_device", "models", "tf_model", "model_float16.tflite")
    fp16_result_path = os.path.join('src', 'edge_device', 'results','orin_fp16.csv')
    df_results, elapsed = run_tflite_inference(model_path, csv_path, fp16_result_path)
    
    model_path = os.path.join("src", "edge_device", "models", "model_int8_full.tflite")
    int8_result_path = os.path.join('src', 'edge_device', 'results','orin_int8.csv')
    df_results, elapsed = run_tflite_inference(model_path, csv_path, int8_result_path)