import argparse
import csv
import logging
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.edge_device.jtop_power_measure import run_jtop_measurement


logger = logging.getLogger(__name__)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="엔진별로 jtop(VDD_IN) 전력 측정을 분리해서 수행합니다")
    p.add_argument("--seconds", type=int, default=120)
    p.add_argument("--rail", type=str, default="VDD_IN")
    p.add_argument("--sample-hz", type=float, default=5.0)
    p.add_argument("--warmup-s", type=float, default=5.0)
    p.add_argument("--device", type=str, default=os.environ.get("DEVICE_TYPE", "orin"))
    p.add_argument(
        "--csv",
        type=str,
        default=os.path.join("src", "train_model", "result", "data", "gn_coords_6.csv"),
    )
    p.add_argument(
        "--out",
        type=str,
        default=os.path.join("src", "edge_device", "results", "power_jtop"),
    )
    p.add_argument(
        "--engines",
        nargs="+",
        default=["idle", "torch", "tflite_fp32", "tflite_fp16", "tflite_int8", "tensorrt"],
        choices=["idle", "torch", "tflite_fp32", "tflite_fp16", "tflite_int8", "tensorrt"],
    )
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--no-kill-after", action="store_true")
    return p.parse_args(argv)


def _workload_cmd(engine: str, device: str, csv_path: str, seconds: int) -> List[str]:
    if engine == "idle":
        # baseline 측정
        return ["sleep", str(int(seconds))]

    return [
        sys.executable,
        "-m",
        "src.edge_device.edge_device",
        "--engine",
        engine,
        "--device",
        device,
        "--csv",
        csv_path,
    ]


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag_base = args.tag or _ts()

    # 실행(run) 단위 폴더: 날짜/시간(tag) + device
    run_dir = out_dir / f"{tag_base}.{args.device}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for engine in args.engines:
        tag = f"{tag_base}.{args.device}.{engine}".replace("/", "_")
        cmd = _workload_cmd(engine, args.device, args.csv, args.seconds)

        logger.info(f"측정 시작: {engine}")
        logger.info(f"workload: {' '.join(shlex.quote(x) for x in cmd)}")

        try:
            summary = run_jtop_measurement(
                cmd=cmd,
                seconds=int(args.seconds),
                out_dir=run_dir / engine,
                tag=tag,
                rail=str(args.rail),
                sample_hz=float(args.sample_hz),
                warmup_s=float(args.warmup_s),
                kill_after=not bool(args.no_kill_after),
            )
        except Exception as e:
            logger.error(f"엔진 측정 실패({engine}): {e}")
            return 1

        stats = summary.get("stats_after_warmup") or {}
        rows.append(
            {
                "tag": tag_base,
                "device": args.device,
                "engine": engine,
                "rail": args.rail,
                "seconds": args.seconds,
                "sample_hz": args.sample_hz,
                "warmup_s": args.warmup_s,
                "avg_w": stats.get("avg_w"),
                "std_w": stats.get("std_w"),
                "p95_w": stats.get("p95_w"),
                "n": stats.get("n"),
                "trace_csv": summary.get("trace_csv"),
            }
        )

    out_csv = run_dir / "summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "tag",
                "device",
                "engine",
                "rail",
                "seconds",
                "sample_hz",
                "warmup_s",
                "avg_w",
                "std_w",
                "p95_w",
                "n",
                "trace_csv",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    logger.info(f"요약 저장: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
