# -*- coding: utf-8 -*-
"""★地球 ── 勾配のある版。★3次元・開いた空間。

★★★2026-09-11 の2回目の書き直し ── **法則を1つ足した**

  ★前の失敗（実測）: ★温度 0.26 → 0.06 と冷え続け、★★**864個全部が1つの塊に凍った**。
     ★スケール 0、★再訪 2。★★★**結晶になった。**
  ★原因: ★**エネルギーの入口と出口が同じ場所だった**（どちらも一様）。
     ★★入口と出口が同じなら流れが無い。★流れが無ければ何も起きない。
  ★★★僕は「重力が潰れて熱くなるから太陽は要らない」と書いたが、★**それが間違い**。
     ★周期の箱では、発散を避けるために平均を引くので、★**全体が潰れることが原理的に起きない**。

★★★法則は**3つ**（★置いた人の指摘で減った）

  ★「勾配」を法則⑤として足そうとしたが、★★**勾配は法則ではない**。
  ★★★にじみ ＝ **局所の差を、隣とのやりとりで均す**。★同じ演算を別の量に当てるだけ:
        ★質量の密度に当てる → ★**重力**
        ★★熱に当てる      → ★★**熱伝導・対流**（★粒子の衝突が既にやっている）
  ★だから足すのは法則ではなく、★★**境界条件**（★中心で入り、外で出る）。
  ★★入口と出口が違う場所にあれば、★**にじみが勝手に勾配を作る**。

   ① ★**ポテンシャル** ── 遠いと引き、近いと反発（LJ）。
                     ★★衝突が熱を運ぶので、★**熱のにじみはここに入っている**
   ② ★**にじみ**   ── 局所の差を隣で均す。★**質量に当てると重力**
   ③ ★**保存**     ── それ以外は何も足さない
   ── ★★境界条件 ── ★★★**中心からエネルギーが入り、外から出る**

★★これで勝手に出てくるはずのもの
   ★丸くなる（重力が中心へ引く）／★層ができる（中心が熱く外が冷たい）／
   ★対流（下が熱く上が冷たい）／★界面（層の境目）／★逃げる（外気の散逸）

★★★最下層にだけ置いた上限
   ★**2種類**。★σ と ε が組み合わせで3通り。★★これが化学の全部。

★★★書いていないもの
   ★分子の一覧・種類の定義・目的関数・報酬・難易度・「くっつくか」の判定・
   ★「これは1つだ」の判定・層の定義・乱数の種

★眺める場所: http://localhost:8766
"""
import hashlib
import os
import sys
import time

import numpy as np

import live

DIM = 3
N0 = int(os.environ.get("EARTH_N", 900))
L = float(os.environ.get("EARTH_L", 44.0))      # ★世界（格子）の広さ
G = int(os.environ.get("EARTH_GRID", 40))
DT = float(os.environ.get("EARTH_DT", 0.004))
R0 = float(os.environ.get("EARTH_R0", 9.0))     # ★はじめの球の半径
R_MAX = L * 0.46                                # ★これより外へ出たら逃げた扱い

P_B = 0.4
SIG = np.array([[1.00, 1.15], [1.15, 1.30]])
EPS = np.array([[1.00, 1.80], [1.80, 0.50]])    # ★★AB が一番強い
RCUT = 2.5

GRAV = float(os.environ.get("EARTH_GRAV", 40.0))
RELAX = 8
# ★★★④勾配 ── 中心から入り、外から出る
R_CORE = float(os.environ.get("EARTH_CORE", 3.5))   # ★ここより内側に注ぐ
POWER = float(os.environ.get("EARTH_POWER", 6.0))   # ★注ぐ仕事率
RAD0 = float(os.environ.get("EARTH_RADIATE", 0.0015))
T_REF = 1.0


class Jitter(object):
    """★★★本物の揺さぶり ── 実行時間のジッタ。★擬似乱数ではない。
    ★どこへ注ぐかを、これが決める。★★再現できないし、周期も無い。"""
    def __init__(self):
        self.buf, self.raw = [], []

    def _tick(self):
        t0 = time.perf_counter_ns()
        s = 0
        for i in range(97):
            s += i * i
        t1 = time.perf_counter_ns()
        return (t1 - t0) ^ (s & 0xff)

    def fill(self, n):
        while len(self.buf) < n:
            d = self._tick()
            self.raw.append(d)
            del self.raw[:-400]
            h = hashlib.blake2b(str(d).encode(), digest_size=16).digest()
            for k in range(0, 16, 2):
                self.buf.append(int.from_bytes(h[k:k + 2], "big") / 65535.0)

    def take(self, n):
        self.fill(n)
        out = np.array(self.buf[:n])
        del self.buf[:n]
        return out

    def health(self):
        if len(self.raw) < 40:
            return 0.0, 0
        a = np.array(self.raw, dtype=float)
        return float(a.std() / max(1.0, a.mean())), len(set(self.raw))


class World(object):
    def __init__(self, jit):
        self.jit = jit
        self.c = np.full(DIM, L / 2.0)
        # ★はじめは中心の球。★格子に置く（★重なると爆発するので）
        pts = []
        side = 26
        for a in range(side):
            for b in range(side):
                for cc in range(side):
                    q = (np.array([a, b, cc]) - (side - 1) / 2.0) * (2.2 * R0 / side)
                    if float(q @ q) < R0 * R0:
                        pts.append(q)
        pts = np.array(pts)
        take = min(N0, len(pts))
        self.pos = self.c + pts[:take]
        self.pos += (jit.take(take * DIM).reshape(take, DIM) - 0.5) * 0.05
        self.n = take
        self.vel = (jit.take(take * DIM).reshape(take, DIM) - 0.5) * 0.8
        self.kind = (jit.take(take) < P_B).astype(int)
        self.m = np.zeros((G,) * DIM)
        self.step_n = 0
        self.seen, self.revisits, self.escaped = {}, 0, 0
        self.pairE = 0.0
        self._mix()

    def _mix(self):
        self.sig = SIG[self.kind[:, None], self.kind[None, :]]
        self.eps = EPS[self.kind[:, None], self.kind[None, :]]

    # ── ② にじみ → 重力（★平均を引かない。★端は吸う）────────
    def field(self):
        gi = tuple(np.clip((self.pos[:, k] / L * G).astype(int), 0, G - 1)
                   for k in range(DIM))
        src = np.zeros((G,) * DIM)
        np.add.at(src, gi, 1.0)
        D = 0.9 / (2.0 * DIM)
        for _ in range(RELAX):
            lp = -2.0 * DIM * self.m
            for ax in range(DIM):
                lp = lp + np.roll(self.m, 1, ax) + np.roll(self.m, -1, ax)
            self.m = self.m + D * lp + src * 0.02
            for ax in range(DIM):                    # ★端は吸う（開いた空間）
                sl = [slice(None)] * DIM
                sl[ax] = 0
                self.m[tuple(sl)] = 0.0
                sl[ax] = -1
                self.m[tuple(sl)] = 0.0
        return gi, np.gradient(self.m)

    # ── ① ポテンシャル（★判定は無い）─────────────────
    def forces(self):
        d = self.pos[:, None, :] - self.pos[None, :, :]
        r2 = (d * d).sum(-1)
        np.fill_diagonal(r2, np.inf)
        s2 = self.sig * self.sig
        cut = r2 < (RCUT * RCUT) * s2
        r2 = np.maximum(r2, 0.7 * s2)
        inv = s2 / r2
        i6 = inv ** 3
        i12 = i6 * i6
        coef = np.where(cut, 24.0 * self.eps * (2.0 * i12 - i6) / r2, 0.0)
        f = (coef[:, :, None] * d).sum(axis=1)
        self.pairE = float(np.where(cut, 4.0*self.eps*(i12-i6), 0.0).sum()) * 0.5
        return f, r2, cut

    def layers(self, r2, cut):
        """★★★空間の勾配が、層を作っているか（★置いた人の指摘 2026-09-11）。

        ★時間で冷える階層は、★**いつか終わる**（冷え切ったら熱的死）。
        ★★空間の勾配は、★**流れがある限り終わらない**。
        ★★★地球の核・マントル・地殻・海・大気は、★**同時に別々の温度で存在している**。

        ★測り方: ★半径で輪切りにして、★**束縛の度合い**の段を数える。
        ★僕は層を定義しない。★段があるかどうかを数えるだけ。
        """
        s2 = self.sig * self.sig
        inv = s2 / np.maximum(r2, 1e-9)
        i6 = inv ** 3
        V = np.where(cut, 4.0 * self.eps * (i6 * i6 - i6), 0.0)
        dv = self.vel[:, None, :] - self.vel[None, :, :]
        bound = cut & ((V + 0.25 * (dv * dv).sum(-1)) < 0.0)
        deg = bound.sum(1).astype(float)
        r = np.sqrt(((self.pos - self.c) ** 2).sum(1))
        edges = np.percentile(r, np.linspace(0, 100, 11))
        prof = []
        for k in range(10):
            sel = (r >= edges[k]) & (r <= edges[k + 1])
            prof.append(float(deg[sel].mean()) if sel.sum() > 4 else 0.0)
        prof = np.array(prof)
        rng = prof.max() - prof.min()
        if rng < 0.3:
            return 1, prof
        steps = int((np.abs(np.diff(prof)) > 0.28 * rng).sum())
        return steps + 1, prof

    def clusters(self, r2, cut):
        """★★束縛 ＝ **対の全エネルギーが負**。★物理の定義。★閾値がゼロ。"""
        s2 = self.sig * self.sig
        inv = s2 / np.maximum(r2, 1e-9)
        i6 = inv ** 3
        V = np.where(cut, 4.0 * self.eps * (i6 * i6 - i6), 0.0)
        dv = self.vel[:, None, :] - self.vel[None, :, :]
        bound = cut & ((V + 0.25 * (dv * dv).sum(-1)) < 0.0)
        ii, jj = np.nonzero(np.triu(bound, 1))
        n = self.n
        par = list(range(n))

        def find(a):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a
        for a, b in zip(ii, jj):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                par[ra] = rb
        cnt = {}
        for i in range(n):
            r = find(i)
            cnt[r] = cnt.get(r, 0) + 1
        return sorted(cnt.values(), reverse=True), len(ii)

    def step(self):
        self.step_n += 1
        f, r2, cut = self.forces()
        gi, grads = self.field()
        for ax in range(DIM):
            f[:, ax] += GRAV * grads[ax][gi]
        rad = self.pos - self.c
        r = np.sqrt((rad * rad).sum(1)) + 1e-9

        # ★★★④勾配 ── 中心に入れる。★どこに入れるかはジッタが決める
        core = np.nonzero(r < R_CORE)[0]
        if len(core):
            k = max(1, len(core) // 3)
            who = core[(self.jit.take(k) * len(core)).astype(int) % len(core)]
            dirv = self.jit.take(k * DIM).reshape(k, DIM) - 0.5
            dirv /= (np.sqrt((dirv*dirv).sum(1))[:, None] + 1e-9)
            # ★注ぐ仕事率を一定にする
            self.vel[who] += dirv * np.sqrt(2.0 * POWER / max(1, k))

        self.vel = self.vel + f * DT
        # ★★★③放射 ── 外側ほど強い（★これが勾配の出口）
        T = float((self.vel * self.vel).sum()) / (DIM * self.n)
        outer = 1.0 + 6.0 * (r / (R_MAX)) ** 2
        damp = 1.0 - RAD0 * (1.0 + (T / T_REF) ** 1.5) * outer
        self.vel *= np.clip(damp, 0.85, 0.9999)[:, None]
        self.pos = self.pos + self.vel * DT

        # ★逃げたものは世界から出る（★大気の散逸。★これも本物）
        rad = self.pos - self.c
        keep = np.sqrt((rad * rad).sum(1)) < R_MAX
        if not keep.all():
            self.escaped += int((~keep).sum())
            self.pos = self.pos[keep]
            self.vel = self.vel[keep]
            self.kind = self.kind[keep]
            self.n = len(self.pos)
            self._mix()
        self._r2, self._cut, self._r = r2, cut, r
        return T

    def stats(self, T):
        sizes, npair = self.clusters(self._r2, self._cut)
        big = sizes[0] if sizes else 0
        multi = sum(1 for s in sizes if s >= 2)
        r = np.sqrt(((self.pos - self.c) ** 2).sum(1))
        v2 = (self.vel * self.vel).sum(1)
        inner = r < np.percentile(r, 25)
        outerm = r > np.percentile(r, 75)
        Tin = float(v2[inner].mean()) / DIM if inner.any() else 0.0
        Tout = float(v2[outerm].mean()) / DIM if outerm.any() else 0.0
        macro = (npair // 12, multi // 4, big // 6, int(T * 8), int(r.mean()))
        h = hash(macro)
        if h in self.seen and self.step_n - self.seen[h] > 40:
            self.revisits += 1
        self.seen[h] = self.step_n
        scales = 0
        a = np.array([s for s in sizes if s >= 2], float)
        if len(a) > 4:
            lb = np.log2(a)
            hist, _ = np.histogram(lb, bins=np.arange(0, lb.max() + 1.5, 1.0))
            scales = int((hist > max(2, 0.04 * len(a))).sum())
        nlay, prof = self.layers(self._r2, self._cut)
        jh, ju = self.jit.health()
        return dict(step=self.step_n, T=round(T, 4), bonds=int(npair),
                    layers=nlay, prof=[round(x, 2) for x in prof],
                    clusters=multi, big=big, scales=scales, n=self.n,
                    revisit=self.revisits, jit=round(jh, 3), juniq=ju,
                    Tin=round(Tin, 4), Tout=round(Tout, 4),
                    grad=round(Tin / max(1e-9, Tout), 2),
                    R=round(float(r.mean()), 2), esc=self.escaped)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    live.start()
    jit = Jitter()
    w = World(jit)
    live.put(running=True, cfg=dict(N=w.n, L=L, dim=DIM))
    live.log("★地球（勾配つき）── ★★法則5つ。★中心から入り、外から出る")
    live.log("★★前の失敗: 一様に入れて一様に出したら**全部凍った**（結晶）")
    live.log("★★★入口と出口が同じなら流れが無い。★流れが無ければ何も起きない")
    hist = {k: [] for k in ("T", "bonds", "clusters", "big", "scales",
                            "revisit", "grad", "R", "n", "layers")}
    t0 = time.time()
    while True:
        T = w.step()
        if w.step_n % 5 == 0:
            st = w.stats(T)
            for k in hist:
                hist[k].append(st[k])
                del hist[k][:-500]
            live.put(stat=dict(st, min=round((time.time() - t0) / 60, 1)),
                     hist=hist, L=L,
                     pos=[[round(float(p[0]), 2), round(float(p[1]), 2),
                           round(float(p[2]), 2)] for p in w.pos],
                     deg=[int(x) for x in w.kind])
        if w.step_n % 100 == 0:
            st = w.stats(T)
            live.log("%5d歩 T%.3f 勾配%.2f ★層%d 半径%.1f 塊%d 最大%d ス%d 再訪%d 逃%d"
                     % (st["step"], st["T"], st["grad"], st["layers"], st["R"],
                        st["clusters"], st["big"], st["scales"], st["revisit"],
                        st["esc"]))
            print("%5d T%.3f 中%.3f/外%.3f 勾配%.2f ★層%d 半径%.1f 対%4d 塊%3d 最大%3d ス%d 再訪%d 逃%d 束縛 %s"
                  % (st["step"], st["T"], st["Tin"], st["Tout"], st["grad"],
                     st["layers"], st["R"], st["bonds"], st["clusters"],
                     st["big"], st["scales"], st["revisit"], st["esc"],
                     st["prof"]), flush=True)


if __name__ == "__main__":
    main()
