import argparse
import csv
import json
import logging
import os
import platform
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

from src.edge_device.watts_sources import extract_rail_watts, list_available_rails


logger = logging.getLogger(__name__)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _terminate_process(proc: subprocess.Popen, grace_s: float = 2.0) -> None:
    if proc.poll() is not None:
        return

    try:
        proc.send_signal(signal.SIGINT)
    except Exception:
        pass

    t0 = time.time()
    while time.time() - t0 < grace_s:
        if proc.poll() is not None:
            return
        time.sleep(0.05)

    try:
        proc.terminate()
    except Exception:
        pass

    t0 = time.time()
    while time.time() - t0 < grace_s:
        if proc.poll() is not None:
            return
        time.sleep(0.05)

    try:
        proc.kill()
    except Exception:
        pass


def _require_jtop():
    try:
        from jtop import jtop  # type: ignore

        return jtop
    except Exception as e:
        raise RuntimeError(
            "jtop import 실패: jetson-stats가 설치되어 있고 jtop.service가 실행 중인지 확인하세요"
        ) from e


def _compute_stats(samples: List[float]) -> Dict[str, Any]:
    if not samples:
        return {"n": 0}

    s = sorted(samples)

    def pct(p: float) -> float:
        if not s:
            return float("nan")
        k = int(round((len(s) - 1) * p))
        return float(s[max(0, min(len(s) - 1, k))])

    return {
        "n": len(samples),
        "avg_w": mean(samples),
        "std_w": pstdev(samples) if len(samples) > 1 else 0.0,
        "min_w": float(min(samples)),
        "max_w": float(max(samples)),
        "p50_w": pct(0.50),
        "p95_w": pct(0.95),
    }


def run_jtop_measurement(
    cmd: List[str],
    seconds: int,
    out_dir: Path,
    tag: str,
    rail: str = "VDD_IN",
    sample_hz: float = 5.0,
    warmup_s: float = 0.0,
    kill_after: bool = True,
) -> Dict[str, Any]:
    """Run workload while sampling jtop power rail.

    - seconds: 최대 실행 시간(초). kill_after=True && seconds>0 일 때만 timeout으로 사용.
      seconds<=0 이면 프로세스가 종료될 때까지 기다립니다.
    - rail: 기본은 Orin에서 흔한 VDD_IN.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    workload_out = out_dir / f"{tag}.workload.stdout.log"
    workload_err = out_dir / f"{tag}.workload.stderr.log"
    trace_csv = out_dir / f"{tag}.jtop.trace.csv"
    summary_json = out_dir / f"{tag}.jtop.summary.json"
    meta_json = out_dir / f"{tag}.jtop.meta.json"

    meta = {
        "tag": tag,
        "seconds": seconds,
        "kill_after": bool(kill_after),
        "rail": rail,
        "sample_hz": sample_hz,
        "warmup_s": warmup_s,
        "cmd": cmd,
        "cmd_str": " ".join(shlex.quote(x) for x in cmd),
        "host": {
            "node": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {"version": sys.version, "executable": sys.executable},
        "time": {"start": datetime.now().isoformat(timespec="seconds")},
    }

    jtop_cls = _require_jtop()

    samples_all: List[float] = []
    samples_after_warmup: List[float] = []
    n_missing = 0

    period = 1.0 / max(0.1, float(sample_hz))

    with workload_out.open("wb") as wf_out, workload_err.open("wb") as wf_err:
        workload_proc = subprocess.Popen(cmd, stdout=wf_out, stderr=wf_err)

        t0 = time.time()
        next_t = t0

        try:
            with jtop_cls() as jetson:
                # 첫 샘플이 준비될 때까지 잠깐 대기
                time.sleep(min(0.5, period))

                with trace_csv.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(
                        f,
                        fieldnames=["t_s", "watts", "rail", "raw_power"],
                    )
                    w.writeheader()

                    while True:
                        now = time.time()
                        if now < next_t:
                            time.sleep(min(period, next_t - now))
                            continue
                        next_t = now + period

                        elapsed = now - t0

                        # workload 종료 체크
                        if workload_proc.poll() is not None:
                            break

                        if kill_after and int(seconds) > 0 and elapsed >= float(seconds):
                            _terminate_process(workload_proc)
                            break

                        stats = getattr(jetson, "stats", None)
                        power = None
                        if isinstance(stats, dict):
                            power = stats.get("power")
                        if power is None and hasattr(jetson, "power"):
                            try:
                                power = getattr(jetson, "power")
                            except Exception:
                                power = None

                        sample = extract_rail_watts(power, rail=rail)
                        if sample is None:
                            n_missing += 1
                            w.writerow({"t_s": f"{elapsed:.3f}", "watts": "", "rail": rail, "raw_power": power})
                            continue

                        samples_all.append(sample.watts)
                        if elapsed >= float(warmup_s):
                            samples_after_warmup.append(sample.watts)

                        w.writerow(
                            {
                                "t_s": f"{elapsed:.3f}",
                                "watts": f"{sample.watts:.6f}",
                                "rail": rail,
                                "raw_power": sample.raw,
                            }
                        )
        except Exception as e:
            msg = str(e)
            if "Mismatch version jtop service" in msg:
                raise RuntimeError(
                    "jtop client/service 버전이 불일치합니다. 아래를 실행한 뒤 다시 시도하세요: sudo systemctl restart jtop.service"
                ) from e
            raise

        workload_rc = workload_proc.wait()

    meta["time"]["end"] = datetime.now().isoformat(timespec="seconds")
    meta["time"]["actual_seconds"] = time.time() - t0
    meta["rc"] = {"workload": workload_rc}

    # rail이 안 잡히면 available rails를 meta에 기록해서 디버깅 도움
    try:
        with jtop_cls() as jetson:
            stats = getattr(jetson, "stats", None)
            power = stats.get("power") if isinstance(stats, dict) else None
            if power is None and hasattr(jetson, "power"):
                try:
                    power = getattr(jetson, "power")
                except Exception:
                    power = None
            meta["available_rails"] = list_available_rails(power)
    except Exception:
        pass

    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    stats_all = _compute_stats(samples_all)
    stats_eff = _compute_stats(samples_after_warmup)

    summary = {
        "tag": tag,
        "rail": rail,
        "sample_hz": sample_hz,
        "warmup_s": warmup_s,
        "n_missing": n_missing,
        "trace_csv": str(trace_csv),
        "workload_rc": workload_rc,
        "stats_all": stats_all,
        "stats_after_warmup": stats_eff,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # rail이 없을 때 메시지를 더 명확히
    if stats_all.get("n", 0) == 0:
        rails = meta.get("available_rails")
        raise RuntimeError(
            f"jtop에서 rail '{rail}' 전력을 읽지 못했습니다. available_rails={rails}. "
            "Jetson Nano에서는 POM_5V_IN 같은 rail을 써야 할 수 있습니다(옵션 --rail)."
        )

    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="jtop으로 Jetson 전력(W)을 측정하고 기록합니다")
    p.add_argument(
        "--seconds",
        type=int,
        default=60,
        help="최대 실행 시간(초). 0 이하이면 타임아웃 없이 프로세스 종료까지 측정합니다",
    )
    p.add_argument(
        "--device",
        type=str,
        default=os.environ.get("DEVICE_TYPE", "nano"),
        help="결과 저장 경로에 사용할 디바이스 태그(기본: $DEVICE_TYPE 또는 nano)",
    )
    p.add_argument(
        "--engine",
        type=str,
        default=None,
        help="엔진 이름(지정 시 out/<tag>.<device>/<engine>/로 저장하고 파일명에도 포함)",
    )
    p.add_argument("--rail", type=str, default="VDD_IN")
    p.add_argument("--sample-hz", type=float, default=5.0)
    p.add_argument("--warmup-s", type=float, default=0.0)
    p.add_argument(
        "--out",
        type=str,
        default=os.path.join("src", "edge_device", "results", "power_jtop"),
        help="결과 저장 베이스 디렉터리(기본: src/edge_device/results/power_jtop)",
    )
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--no-kill-after", action="store_true")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    if not args.cmd or args.cmd == ["--"]:
        logger.error("실행할 커맨드를 `--` 뒤에 넣어주세요. 예) -- python -m src.edge_device.edge_device --engine torch")
        return 2

    cmd = args.cmd
    if cmd[0] == "--":
        cmd = cmd[1:]

    tag_base = str(args.tag or _ts()).replace("/", "_")
    device = str(args.device).replace("/", "_")
    engine = str(args.engine).replace("/", "_") if args.engine else None

    out_base = Path(args.out)
    run_dir = out_base / f"{tag_base}.{device}"
    out_dir = (run_dir / engine) if engine else run_dir

    # 파일명 tag는 기존과 동일하게 <tag>.<device>[.<engine>]
    tag = f"{tag_base}.{device}" + (f".{engine}" if engine else "")

    kill_after = (not bool(args.no_kill_after)) and int(args.seconds) > 0
    seconds_desc = str(args.seconds) if kill_after else "until-exit"
    logger.info(
        f"jtop 측정 시작 rail={args.rail} seconds={seconds_desc} sample_hz={args.sample_hz} out={out_dir}"
    )

    try:
        summary = run_jtop_measurement(
            cmd=cmd,
            seconds=int(args.seconds),
            out_dir=out_dir,
            tag=tag,
            rail=str(args.rail),
            sample_hz=float(args.sample_hz),
            warmup_s=float(args.warmup_s),
            kill_after=kill_after,
        )
    except Exception as e:
        logger.error(str(e))
        return 1

    # benchmark처럼 run 폴더에 summary.csv를 남김(단일 엔진/커스텀 workload도 기록)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        out_csv = run_dir / "summary.csv"
        stats = summary.get("stats_after_warmup") or {}
        row = {
            "tag": tag_base,
            "device": device,
            "engine": engine or "custom",
            "rail": str(args.rail),
            "seconds": int(args.seconds),
            "sample_hz": float(args.sample_hz),
            "warmup_s": float(args.warmup_s),
            "avg_w": stats.get("avg_w"),
            "std_w": stats.get("std_w"),
            "p95_w": stats.get("p95_w"),
            "n": stats.get("n"),
            "trace_csv": summary.get("trace_csv"),
        }
        fieldnames = [
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
        ]
        write_header = (not out_csv.exists()) or out_csv.stat().st_size == 0
        with out_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        logger.warning(f"summary.csv 저장 실패(무시): {e}")

    logger.info("완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
