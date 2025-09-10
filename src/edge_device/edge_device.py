import os
import torch

from src.edge_device.tflite import run_tflite_inference
from src.edge_device.pytorch import run_pytorch_inference
from utils.tools import createDirectory
    
DEVICE_TYPE = 'rbp4'
csv_path = os.path.join('src', 'train_model', 'result', 'data', 'gn_coords_6.csv')
result_path = os.path.join('src', 'edge_device', 'results', DEVICE_TYPE)
createDirectory(result_path)

if __name__ == "__main__":
    
    # CUDA 컨텍스트 초기화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # PyTorch 모델 사용
    model_path = os.path.join('src', 'train_model', 'result', 'models', 'num_gu', 'best_num_gu_6.pt')
    csv_path = os.path.join('src', 'train_model', 'result', 'data', 'gn_coords_6.csv')
    
    df_results, elapsed = run_pytorch_inference(
        model_path, 
        csv_path, 
        use_batch=False, 
        batch_size=1,
        output_path=os.path.join(result_path, f'{DEVICE_TYPE}_torch.csv')
    )
    
    # TFLite_FP32 모델 사용
    tflite32_model_path = os.path.join("src", "edge_device", "models", "tf_model", "model_float32.tflite")
    output_path = os.path.join(result_path, f'{DEVICE_TYPE}_tflite.csv')

    df_results, elapsed = run_tflite_inference(
        tflite32_model_path,
        csv_path,
        output_path=output_path
    )
    
    # TFLite_FP16 모델 사용
    tflite16_model_path = os.path.join("src", "edge_device", "models", "tf_model", "model_float16.tflite")
    output_path = os.path.join(result_path, f'{DEVICE_TYPE}_tflite_fp16.csv')

    df_results, elapsed = run_tflite_inference(
        tflite16_model_path,
        csv_path,
        output_path=output_path
    )
    
    # TFLite_INT8 모델 사용
    tflite_int8_model_path = os.path.join("src", "edge_device", "models", "model_int8_full.tflite")
    output_path = os.path.join(result_path, f'{DEVICE_TYPE}_tflite_int8.csv')
    df_results, elapsed = run_tflite_inference(
        tflite_int8_model_path,
        csv_path,
        output_path=output_path
    )
    
    # TensorRT 모델 사용
    if torch.cuda.is_available():
        from src.edge_device.tensorRT import run_tensorrt_inference
        
        engine_path = os.path.join("src", "edge_device", "models", "model_fp16.engine")
        output_path = os.path.join(result_path, f'{DEVICE_TYPE}_tensorRT.csv')
        df_results, elapsed = run_tensorrt_inference(engine_path, csv_path, output_path)
    