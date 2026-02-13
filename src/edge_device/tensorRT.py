import os
import time
import logging
from typing import Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import torch
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except Exception as e:
    if e is ImportError:
        logging.warning("TensorRT or PyCUDA not available. Please install them for GPU inference.");
    
    logging.warning(f"Error details: {e}")
    trt = None
    cuda = None

from utils.config import Config
from tqdm import trange

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Constants
COORD_SCALE = 200
COORD_OFFSET = 100
DEFAULT_Z = 70


class TRTEngineInfo:
    """TensorRT 엔진 정보"""
    def __init__(self, engine):
        self.engine = engine
        self.context = engine.create_execution_context()
        
        # 입출력 바인딩 정보
        self.inp_name = None
        self.out_name = None
        self.inp_shape = None
        self.out_shape = None
        self.inp_dtype = None
        self.out_dtype = None
        
        # TensorRT 8.x 호환을 위한 API 사용
        if hasattr(engine, 'num_io_tensors'):
            # TensorRT 8.x+ 새로운 API
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                shape = engine.get_tensor_shape(name)
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                mode = engine.get_tensor_mode(name)
                
                if mode == trt.TensorIOMode.INPUT:
                    self.inp_name = name
                    self.inp_shape = shape
                    self.inp_dtype = dtype
                else:
                    self.out_name = name
                    self.out_shape = shape
                    self.out_dtype = dtype
        else:
            # TensorRT 7.x 이전 API (deprecated)
            for i in range(engine.num_bindings):
                name = engine.get_binding_name(i)
                shape = engine.get_binding_shape(i)
                dtype = trt.nptype(engine.get_binding_dtype(i))
                
                if engine.binding_is_input(i):
                    self.inp_name = name
                    self.inp_shape = shape
                    self.inp_dtype = dtype
                else:
                    self.out_name = name
                    self.out_shape = shape
                    self.out_dtype = dtype


class TRTEngine:
    """TensorRT 추론 엔진"""
    
    def __init__(self, engine_path: str):
        if trt is None:
            raise ImportError("TensorRT not available")
            
        # TensorRT 런타임 초기화
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # 엔진 로드
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.info = TRTEngineInfo(self.engine)
        
        # GPU 메모리 할당
        self._alloc_buffers()
    
    def _alloc_buffers(self):
        """GPU/CPU 버퍼 할당"""
        self.h_inp = cuda.pagelocked_empty(
            trt.volume(self.info.inp_shape), dtype=self.info.inp_dtype
        )
        self.h_out = cuda.pagelocked_empty(
            trt.volume(self.info.out_shape), dtype=self.info.out_dtype
        )
        
        self.d_inp = cuda.mem_alloc(self.h_inp.nbytes)
        self.d_out = cuda.mem_alloc(self.h_out.nbytes)
        
        self.bindings = [int(self.d_inp), int(self.d_out)]
        self.stream = cuda.Stream()
    
    def _warmup(self, data: np.ndarray) -> None:
        """워밍업"""
        sample = data[0:1].flatten().astype(self.info.inp_dtype)
        np.copyto(self.h_inp, sample)
        
        cuda.memcpy_htod_async(self.d_inp, self.h_inp, self.stream)
        self.info.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()
    
    def _prep_inp(self, data: np.ndarray) -> np.ndarray:
        """입력 전처리"""
        return data.flatten().astype(self.info.inp_dtype)
    
    def _proc_out(self, raw: np.ndarray) -> np.ndarray:
        """출력 후처리"""
        return raw.reshape(-1, self.info.out_shape[-1])
    
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """배치 추론"""
        n_samples = data.shape[0]
        
        # 워밍업
        self._warmup(data)
        
        # 결과 배열 할당
        out_size = trt.volume(self.info.out_shape)
        results = np.empty((n_samples, out_size), dtype=np.float32)
        
        start = time.perf_counter()
        
        for i in trange(n_samples, desc="TensorRT Inference", leave=False):
            # 입력 준비
            inp = self._prep_inp(data[i:i+1])
            np.copyto(self.h_inp, inp)
            
            # GPU 추론
            cuda.memcpy_htod_async(self.d_inp, self.h_inp, self.stream)
            self.info.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
            self.stream.synchronize()
            
            # 결과 저장
            results[i] = self.h_out[:out_size]
        
        elapsed = time.perf_counter() - start
        return results, elapsed
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'd_inp'):
            self.d_inp.free()
        if hasattr(self, 'd_out'):
            self.d_out.free()


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


def run_tensorrt_inference(engine_path: str, csv_path: str, 
                          output_path: Optional[str] = None) -> Tuple[pd.DataFrame, float]:
    """TensorRT 추론 실행"""
    cfg = Config(num_users=6)
    orig_df, data = load_data(csv_path, cfg)
    
    engine = TRTEngine(engine_path)
    preds, elapsed = engine.predict(data)
    
    results_df = make_results(orig_df, preds, elapsed)
    
    if output_path is None:
        output_path = os.path.join('src', 'edge_device', 'results_tensorrt.csv')
    
    results_df.to_csv(output_path, index=False)
    print_stats(len(data), elapsed, output_path)
    
    return results_df, elapsed


if __name__ == "__main__":
    # 기본 실행
    engine_path = os.path.join("src", "edge_device", "models", "model_fp16.engine")
    csv_path = os.path.join('src', 'train_model', 'result', 'data', 'gn_coords_6.csv')
    
    df_results, elapsed = run_tensorrt_inference(engine_path, csv_path)
