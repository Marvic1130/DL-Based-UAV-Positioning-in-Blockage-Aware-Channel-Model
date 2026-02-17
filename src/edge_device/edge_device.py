import argparse
import logging
import os
from typing import Optional

import cv2
import torch

from src.edge_device.pytorch import run_pytorch_inference
from src.edge_device.tflite import run_tflite_inference
from utils.tools import createDirectory


logger = logging.getLogger(__name__)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run edge inference engines")
    p.add_argument(
        "--engine",
        type=str,
        default="all",
        choices=["all", "torch", "tflite_fp32", "tflite_fp16", "tflite_int8", "tensorrt"],
        help="실행할 엔진(기본 all)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=os.environ.get("DEVICE_TYPE", "nano"),
        help="결과 저장 폴더 이름(기본 $DEVICE_TYPE 또는 nano)",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=os.path.join("src", "train_model", "result", "data", "gn_coords_6.csv"),
        help="입력 CSV 경로",
    )
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    device_type = str(args.device)
    csv_path = str(args.csv)

    result_path = os.path.join("src", "edge_device", "results", device_type)
    createDirectory(result_path)

    def run_tensorrt() -> None:
        if not torch.cuda.is_available():
            logger.warning("CUDA not available; skip TensorRT")
            return

        try:
            from src.edge_device.tensorRT import run_tensorrt_inference
        except Exception as e:
            logger.warning(f"TensorRT import 실패(스킵): {e}")
            return

        # CUDA 컨텍스트 초기화
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        engine_path = os.path.join("src", "edge_device", "models", "model_fp16.engine")
        output_path = os.path.join(result_path, f"{device_type}_tensorRT.csv")
        run_tensorrt_inference(engine_path, csv_path, output_path)

    def run_torch() -> None:
        model_path = os.path.join(
            "src",
            "train_model",
            "result",
            "models",
            "num_gu",
            "best_num_gu_6.pt",
        )
        run_pytorch_inference(
            model_path,
            csv_path,
            use_batch=False,
            batch_size=1,
            output_path=os.path.join(result_path, f"{device_type}_torch.csv"),
        )

    def run_tflite_fp32() -> None:
        model_path = os.path.join("src", "edge_device", "models", "tf_model", "model_float32.tflite")
        output_path = os.path.join(result_path, f"{device_type}_tflite.csv")
        run_tflite_inference(model_path, csv_path, output_path=output_path)

    def run_tflite_fp16() -> None:
        model_path = os.path.join("src", "edge_device", "models", "tf_model", "model_float16.tflite")
        output_path = os.path.join(result_path, f"{device_type}_tflite_fp16.csv")
        run_tflite_inference(model_path, csv_path, output_path=output_path)

    def run_tflite_int8() -> None:
        model_path = os.path.join("src", "edge_device", "models", "model_int8_full.tflite")
        output_path = os.path.join(result_path, f"{device_type}_tflite_int8.csv")
        run_tflite_inference(model_path, csv_path, output_path=output_path)

    engine = str(args.engine)
    if engine in ("all", "tensorrt"):
        run_tensorrt()
    if engine in ("all", "torch"):
        run_torch()
    if engine in ("all", "tflite_fp32"):
        run_tflite_fp32()
    if engine in ("all", "tflite_fp16"):
        run_tflite_fp16()
    if engine in ("all", "tflite_int8"):
        run_tflite_int8()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())