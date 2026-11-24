# numpy 배열, Path 등을 포함한 결과를 JSON으로 안전하게 저장
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path

import numpy as np


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple, deque)):
        return list(value)
    return str(value)


# 임시 파일에 쓴 뒤 교체 (중간에 중단돼도 깨진 JSON이 남지 않음)
def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    text = json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)
