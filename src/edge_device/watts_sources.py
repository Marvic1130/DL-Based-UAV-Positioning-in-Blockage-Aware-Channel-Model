import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class WattsSample:
    watts: float
    raw: Any


def _parse_watts_string(s: str) -> Optional[float]:
    txt = s.strip()
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(mW|W)\b", txt, flags=re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "mw":
        return val / 1000.0
    return val


def _coerce_watts(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        v = float(value)
        # jtop에서 mW 정수로 오는 케이스가 흔해서 heuristic 적용
        # (Jetson 보드 입력 전력은 보통 1~50W 사이)
        if v > 200.0:
            return v / 1000.0
        return v

    if isinstance(value, str):
        return _parse_watts_string(value)

    return None


def extract_rail_watts(power: Any, rail: str = "VDD_IN") -> Optional[WattsSample]:
    """jtop stats의 power 필드에서 특정 rail의 전력(W)을 추출합니다.

    power는 보통 dict이며, 레일별 값이 다음 중 하나 형태로 옵니다:
    - number (W 또는 mW)
    - string (예: "3.2 W", "950 mW")
    - dict (예: {"power": 3500, ...})
    """
    if not isinstance(power, dict):
        return None

    # jtop 버전에 따라 power 구조가 다름.
    # (A) stats['power'] 형태: {'VDD_IN': {...}, 'VDD_SOC': {...}, ...}
    # (B) jetson.power 형태: {'rail': {'VDD_SOC': {...}}, 'tot': {..., 'name': 'VDD_IN'}}
    if "tot" in power and isinstance(power.get("tot"), dict):
        tot = power.get("tot")
        name = str(tot.get("name", ""))
        if name == rail:
            rail_obj = tot
        else:
            rail_obj = None
            rails = power.get("rail")
            if isinstance(rails, dict):
                rail_obj = rails.get(rail)
    else:
        rail_obj = power.get(rail)

    if rail_obj is None:
        return None

    if isinstance(rail_obj, dict):
        # 가장 흔한 키들 우선순위
        for key in ("power", "avg", "cur", "value"):
            w = _coerce_watts(rail_obj.get(key))
            if w is not None:
                return WattsSample(watts=w, raw=rail_obj)
        return None

    w = _coerce_watts(rail_obj)
    if w is None:
        return None
    return WattsSample(watts=w, raw=rail_obj)


def list_available_rails(power: Any) -> Tuple[str, ...]:
    if not isinstance(power, dict):
        return tuple()
    if "tot" in power and isinstance(power.get("tot"), dict):
        rails = []
        tot = power.get("tot")
        name = tot.get("name")
        if name:
            rails.append(str(name))
        rail_map = power.get("rail")
        if isinstance(rail_map, dict):
            rails.extend(str(k) for k in rail_map.keys())
        return tuple(sorted(set(rails)))
    return tuple(sorted(str(k) for k in power.keys()))
