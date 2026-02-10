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

    - seconds: 최대 실행 시간(초). kill_after=True일 때 timeout으로 사용.
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

                        if kill_after and elapsed >= float(seconds):
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
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--rail", type=str, default="VDD_IN")
    p.add_argument("--sample-hz", type=float, default=5.0)
    p.add_argument("--warmup-s", type=float, default=0.0)
    p.add_argument(
        "--out",
        type=str,
        default=os.path.join("src", "edge_device", "results", "power_jtop"),
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

    tag = args.tag or _ts()
    out_dir = Path(args.out)

    logger.info(f"jtop 측정 시작 rail={args.rail} seconds={args.seconds} sample_hz={args.sample_hz}")

    try:
        run_jtop_measurement(
            cmd=cmd,
            seconds=int(args.seconds),
            out_dir=out_dir,
            tag=tag,
            rail=str(args.rail),
            sample_hz=float(args.sample_hz),
            warmup_s=float(args.warmup_s),
            kill_after=not bool(args.no_kill_after),
        )
    except Exception as e:
        logger.error(str(e))
        return 1

    logger.info("완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
