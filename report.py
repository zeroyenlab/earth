# -*- coding: utf-8 -*-
"""★いまの地球を1行で言う道具。★歩数だけ欲しいときは step を渡す。

★★workflow の中に python -c の本文を直接書くと、★**YAML の塊が0列で割れる**。
　★だから外に出した（2026-09-11 に一度割った）。
"""
import io
import json
import os
import sys

P = os.environ.get("EARTH_HIST", "history.json")
if not os.path.exists(P):
    print("0" if len(sys.argv) > 1 and sys.argv[1] == "step" else "まだ無い")
    raise SystemExit(0)

d = json.load(io.open(P, encoding="utf-8"))
if len(sys.argv) > 1 and sys.argv[1] == "step":
    print(int(d.get("step", 0)))
    raise SystemExit(0)

s = d.get("stat", {})
print("%s歩 種類%s 分子%s 層%s 粒%s ずれ%s%%"
      % (d.get("step"), s.get("kinds"), s.get("mol"),
         s.get("layers"), s.get("n"), s.get("drift")))
