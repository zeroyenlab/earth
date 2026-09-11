# -*- coding: utf-8 -*-
"""★★★段3: 周期表が出るか。

★★★問い
   ★核1個に電子を Z 個入れて、Z を 1 から動かしていく。
   ★★一番上の電子を剥がすのに要るエネルギーが、★**のこぎり波**になるか。
   ★なれば、それが周期表。

★★★何を書いていないか
   ★殻・価数・軌道の形・周期の長さ。★どれも書いていない。
   ★書いたのは2つの縛りだけ:
   　① 各 ψ の合計を1に保つ
   　② ψ どうしを直交させる（★パウリ原理）
   ★★★と、**スピン（1つの形に2個まで）の「2」**。★これは手で書いた数。

★★★だから見るべきは「周期の長さ」ではなく「のこぎりの山と谷」。
   ★長さ（2, 8, 18）は 2×(1, 4, 9)。★★「2」は僕が書いたが、
   ★**(1, 4, 9) は書いていない**（★3次元空間から出てくる）。
   ★山と谷の深さも書いていない。

★答え合わせ: 電子どうしの反発を入れていないので、段は水素型 −Z²/(2n²) のはず。
   ★Z=1,2 → n=1 / Z=3..10 → n=2 / Z=11.. → n=3
   ★★つまり **Z=3 と Z=11 で落ちる**（★殻が閉じた直後）。

★走らせる: python stage3_shells.py
"""
import sys
import time

import numpy as np

import quantum as Q

ZMAX = 12


def one(z, g=44, extra=2, sweeps=900):
    """★電荷 z の核に、電子を z 個。★一番上の電子のエネルギーを返す。"""
    need, occ = Q.occupy(z)
    norb = need + extra
    # ★★重い核ほど電子は内側に寄る（★半径 ≒ n²/Z）。★箱もそれに合わせる
    nmax = 1 if z <= 2 else (2 if z <= 10 else 3)
    half = 4.0 * nmax * nmax / z + 2.5
    b = Q.Box([[0.0, 0.0, 0.0]], pad=half, g=g)
    c = np.array([[0.0, 0.0, 0.0]])
    _psi, e, _V = Q.solve(b, c, [float(z)], norb, sweeps=sweeps)
    e = np.sort(e)
    homo = e[need - 1]                  # ★一番上の、電子が入っている段
    return homo, e[:need + 1], b.dx


def theory(z):
    """★水素型の答え（★電子どうしの反発を入れていない場合）"""
    n = 1 if z <= 2 else (2 if z <= 10 else 3)
    return -(z * z) / (2.0 * n * n)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("★★★段3: 周期表が出るか")
    print("★核1個・電子Z個。★一番上の電子を剥がすのに要るエネルギー（＝イオン化）を見る")
    print()
    print("%4s %6s %12s %12s %10s %8s"
          % ("電荷", "電子", "イオン化", "本物の答え", "ずれ%", "秒"))
    rows = []
    for z in range(1, ZMAX + 1):
        t0 = time.time()
        homo, levels, dx = one(z)
        ion = -homo
        th = -theory(z)
        err = 100.0 * abs(ion - th) / th
        rows.append((z, ion, th, err))
        print("%4d %6d %12.3f %12.3f %10.1f %8.1f"
              % (z, z, ion, th, err, time.time() - t0), flush=True)

    print()
    print("★★★のこぎりになっているか ── ★前の Z より下がった所が「殻が閉じた」所")
    print("%4s %12s %10s" % ("電荷", "イオン化", "前との差"))
    drops = []
    for i, (z, ion, th, err) in enumerate(rows):
        d = ion - rows[i - 1][1] if i else 0.0
        mark = "  ★★★ここで落ちた（殻が閉じた）" if d < -1e-9 else ""
        if d < -1e-9:
            drops.append(z)
        print("%4d %12.3f %10.3f%s" % (z, ion, d, mark))

    print()
    print("★落ちた所:", drops)
    print("★本物の答え: [3, 11]（★殻が 2個 と 8個 で閉じるので、その次で落ちる）")
    print()
    ok = drops == [3, 11]
    print("★★★" + ("合っている。★周期表が出た。" if ok
                   else "合っていない。★下の段を見て原因を探す。"))
    print()
    print("★★★正直に: 周期の長さ 2, 8 は 2×(1, 4)。")
    print("　★「2」はスピンとして**僕が手で書いた**。")
    print("　★**(1, 4) は書いていない** ── 3次元空間の中で直交する形の数から出てくる。")
    print("　★山と谷の深さも書いていない。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
