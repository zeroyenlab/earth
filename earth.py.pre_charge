# -*- coding: utf-8 -*-
"""★地球 v2 ── 書き直し（2026-09-11）

★★★v1 で分かったこと（★全部実測か、★置いた人の指摘）
   ★① 一様に入れて一様に出すと**凍る**（★実測: 864個が1つに、スケール0、再訪2）
      → ★**入口と出口は違う場所**
   ★② 仕事率をつまみにするのは**外の手**（★置いた人）
      → ★**熱源は世界の中から出なければならない**
   ★③ 種類が2つ固定だと、★★**原理②（組み合わせが次の素になる）を満たしていない**
      → ★**種類が増える仕組みが要る**

★★★そして ②と③ は**同じ1つの法則で解ける**

   ★ケルヴィンは「地球は収縮熱だけなら数千万年で冷える」と計算して地質学と矛盾した。
   ★★解決したのは**放射能の発見**。★★★**終わらない熱源 ＝ 種類が変わること**。

★★法則（★4つ）
   ① ★**ポテンシャル** ── 遠いと引き、近いと反発（LJ）。★大きさは質量から（σ ∝ m^⅓）
   ② ★**にじみ**     ── 局所の差を隣で均す。★質量に当てると**重力**（★3次元で 1/r²・実測）
   ③ ★★★**変換**   ── ★軽いものは**融合**して熱を出す。★重いものは**分裂**して熱を出す。
                    ★★**どちらが起きるかは、結合エネルギーの曲線だけが決める**
   ④ ★**保存**       ── 外へ出た熱だけが世界から消える（★放射）

★★★僕が最下層に置いた上限 ── **結合エネルギーの曲線、1本だけ**
   ★B(m)/m が、ある質量で最大になる山型。★★**その質量が何になるかは、僕は決めない**。
   ★★★これ1本で「軽いものは融合し、重いものは分裂し、真ん中に溜まる」が全部決まる
      （★実際の宇宙で鉄が一番安定なのと同じ理由）。

★★★書いていないもの
   ★元素の一覧・種類の定義・仕事率・目的関数・報酬・難易度・層の定義・
   ★「くっつくか」の判定・「これは1つだ」の判定・乱数の種

★★★崩壊の「いつ」は、★**本物の実行時間のジッタ**が決める。
   ★実際の放射性崩壊も真にランダムで、★予測できない。★そこが一番自然な入口。

★眺める場所: http://localhost:8766
"""
import hashlib
import io
import json
import math
import os
import sys
import time

import numpy as np

import live

DIM = 3
N0 = int(os.environ.get("EARTH_N", 700))
L = float(os.environ.get("EARTH_L", 48.0))
G = int(os.environ.get("EARTH_GRID", 40))
DT = float(os.environ.get("EARTH_DT", 0.004))
R0 = float(os.environ.get("EARTH_R0", 8.0))
R_MAX = L * 0.46

# ── ★★★続き ── PCを閉じても、機械が変わっても、同じ世界が続く ──────
#   ★世界は「いま」しか持っていない。★その「いま」を丸ごと紙に落とす。
#   ★★続きから読むと、**同じ場所・同じ速さ・同じ台帳**で再開する。
#   ★★★再訪の指紋（trail/seen）も持ち越す。★これが無いと「新しさ」が毎回1.00に戻り、
#      ★**周期に入っていても気づけない**。
HERE = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.environ.get("EARTH_STATE", os.path.join(HERE, "state.npz"))
HIST_PATH = os.environ.get("EARTH_HIST", os.path.join(HERE, "history.json"))
RESUME = os.environ.get("EARTH_RESUME", "1") != "0"
MAX_STEPS = int(os.environ.get("EARTH_STEPS", 0))       # ★0 = 止めずにずっと
MAX_MIN = float(os.environ.get("EARTH_MINUTES", 0))     # ★0 = 時間で区切らない
SAVE_EVERY = int(os.environ.get("EARTH_SAVE_EVERY", 500))

HIST_KEYS = ("T", "kinds", "mmax", "layers", "grad", "R", "n",
             "revisit", "fuse", "fiss", "Ebind", "Ekin", "Eout",
             "shannon", "mscales", "newness", "fuel", "drift",
             "mol", "molmax", "molkinds", "bonds")

# ── ★★★最下層に置いた唯一の上限: 結合エネルギーの曲線 ──────
M_PEAK = float(os.environ.get("EARTH_MPEAK", 8.0))   # ★一番安定になる質量
E0 = float(os.environ.get("EARTH_E0", 2.0))          # ★深さ
S_W = 1.2                                            # ★山の広さ


def binding(m):
    """★★★結合エネルギー。★これ1本で融合も分裂も決まる。

    ★B(m)/m が M_PEAK で最大になる山。
    ★★軽い同士の融合は発熱、★重いものの分裂は発熱、★その逆は起きない。
    ★★★僕は「何が安定か」を書いていない。★曲線が決める。
    """
    x = np.log(np.maximum(m, 1e-9) / M_PEAK)
    return E0 * m * np.exp(-(x * x) / (2.0 * S_W * S_W))


# ── ★★★名前 ────────────────────────────────
#   ★この世界では**質量そのものが種類**なので、★名前も質量から機械的に作る。
#   ★★これは実際の IUPAC が**まだ名前の無い元素**に使うやり方と同じ
#     （★数字を桁ごとに読む。★例: 118番 = ウンウンオクチウム = Uuo）。
#   ★★★だから「人間の周期表」は一切入っていない。★質量8を酸素とは呼ばない
#     （★そもそも中身が違う。★ここの8は質量であって陽子の数ではない）。
_DIG = ("Nil", "Un", "Bi", "Tri", "Quad", "Pent", "Hex", "Sept", "Oct", "Enn")
_SUB = "₀₁₂₃₄₅₆₇₈₉"


def sym(m):
    """★質量 → 名前。★例: 8→Oct  12→UnBi  21→BiUn"""
    return "".join(_DIG[int(c)] for c in str(int(m)))


def formula(f):
    """★中身 → 式。★例: (8,8)→Oct₂  (6,8)→HexOct  (8,8,10)→Oct₂UnNil"""
    out, i = [], 0
    f = sorted(f)
    while i < len(f):
        j = i
        while j < len(f) and f[j] == f[i]:
            j += 1
        k = j - i
        out.append(sym(f[i]) + ("" if k == 1 else
                                "".join(_SUB[int(c)] for c in str(k))))
        i = j
    # ★★「・」で区切る。★UnNilUnUn だと「10と11」か読めない（2026-09-11）
    return "・".join(out)


SIG0 = 1.0
# ★★★化学の強さ ── **曲線をもう一度使う**（2026-09-11）。★新しい表は書かない。
#   ★実際の化学では、★**閉殻に近い元素ほど結びつかない**（ヘリウム・ネオン）。
#   ★★離れているほど反応する。
#   ★僕は既に「安定さの曲線」B(m)/m を持っている。★**山の近く＝安定＝不活性**。
#   ★★★これで**希ガスが勝手に出る**。★僕は「何が不活性か」を書いていない。
#   ★（★核の安定さと化学の不活性さは、実際には別の理屈。★ここは**似せている**と正直に言う）
EPS_MIN = float(os.environ.get("EARTH_EPSMIN", 0.05))
EPS_MAX = float(os.environ.get("EARTH_EPSMAX", 1.2))


def reactivity(m):
    """★0 = 山の上（不活性） 〜 1 = 山から遠い（反応する）"""
    return 1.0 - binding(m) / np.maximum(1e-9, E0 * m)
RCUT = 2.5
GRAV = float(os.environ.get("EARTH_GRAV", 40.0))
RELAX = 8
SRC_K = float(os.environ.get("EARTH_SRCK", 0.06))   # ★源の強さ
RAD0 = float(os.environ.get("EARTH_RADIATE", 0.0015))
T_REF = 1.0
# ★★★クーロン障壁（2026-09-11）。★前は**定数 1.2** だった。
#   ★実際の核融合が鉄で止まる理由は2つあって、★僕は片方（結合エネルギーの曲線）しか
#   ★入れていなかった。★★もう1つが**電気の反発**で、★**重いもの同士ほど近づけない**。
#   ★障壁 ∝ 電荷×電荷 ÷ 半径 ≒ (ma·mb) / (ma^⅓ + mb^⅓)
#   ★★★太陽が水素を燃やせて鉄を燃やせないのは、これが理由。
#   ★つまみは増えない（★定数1つが式1つに変わるだけ）。
COULOMB = float(os.environ.get("EARTH_COULOMB", 0.55))
FISSION_P = float(os.environ.get("EARTH_FISSION", 0.004))    # ★1歩あたりの崩壊しやすさ


class Jitter(object):
    """★★★本物の揺さぶり ── 実行時間のジッタ。★擬似乱数ではない。
    ★★崩壊がいつ起きるかを、これが決める。★再現できないし、周期も無い。"""
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


class Earth(object):
    def __init__(self, jit):
        self.jit = jit
        self.c = np.full(DIM, L / 2.0)
        pts = []
        side = 24
        for a in range(side):
            for b in range(side):
                for cc in range(side):
                    q = (np.array([a, b, cc]) - (side - 1) / 2.0) * (2.3 * R0 / side)
                    if float(q @ q) < R0 * R0:
                        pts.append(q)
        pts = np.array(pts)
        n = min(N0, len(pts))
        self.pos = self.c + pts[:n] + (jit.take(n * DIM).reshape(n, DIM) - 0.5) * 0.05
        self.vel = (jit.take(n * DIM).reshape(n, DIM) - 0.5) * 1.0
        self.mass = np.ones(n)                    # ★★全部「1」から始まる。★水素だけ
        # ★★★名札。★「同じ分子が壊れずに続いているか」を測るのに要る。
        #   ★中身（質量の並び）だけでは、★**別の粒で出来た別の分子**と区別できない。
        self.pid = np.arange(n, dtype=np.int64)
        self.next_pid = int(n)
        self.m3 = np.zeros((G,) * DIM)
        self.step_n = 0
        self.secs = 0.0          # ★★★この世界が生きてきた合計の秒数（★区切りをまたぐ）
        self.seen, self.revisits, self.escaped = {}, 0, 0
        self.trail = []
        self.nfuse = self.nfiss = 0
        self.nfuse_endo = 0
        self.heat = 0.0
        self.release = []
        self.Ein = self.Eout = 0.0
        # ★★★台帳 ── 質量とエネルギーが、どこへ行ったか
        self.mass0 = float(self.mass.sum())
        # ★★燃料の満タン ＝ 全部が山の質量になったときの結合エネルギー
        self.Ebind_max = float(binding(np.array([M_PEAK]))[0]) * self.mass0 / M_PEAK
        self.mass_esc = 0.0
        self.E0tot = None
        # ★★★診断: どの操作でいくら湧いたか
        self.d_move = self.d_tr = self.d_del = 0.0
        self.Wgrav = self._wprev = 0.0
        self.reach2 = 36.0
        self.Elj = 0.0
        # ★★★図鑑 ── **これまでに出来たもの**。★分子は数歩で消えるので、
        #   ★「いま」だけ見せると、開いた瞬間によっては**何も無い**ことになる。
        #   ★★出た瞬間を覚えておく。★消えても、出たことは残る。
        self.zukan_a = {}     # ★質量 → [はじめて出た歩, 見た回数]
        self.zukan_m = {}     # ★中身 → [はじめて出た歩, 見た回数, 最長何歩続いたか]
        self.mol_run = {}     # ★★★いま何歩続いているか（★これが「消えない」の本体）

    def new_pid(self):
        self.next_pid += 1
        return self.next_pid - 1

    @property
    def n(self):
        return len(self.pos)

    def eps(self):
        """★★結びつきの強さ。★★★質量から決まる。★僕は表を書かない。"""
        e = EPS_MIN + (EPS_MAX - EPS_MIN) * reactivity(self.mass)
        return np.sqrt(e[:, None] * e[None, :])       # ★混ぜ方は標準の相乗平均

    def sig(self):
        """★★★大きさは質量に依らない（2026-09-11 の訂正）。

        ★前は σ ∝ m^⅓ にしていた。★融合すると**大きさが1.26倍に跳ねて隣と重なり**、
        ★★LJ が爆発して、★**払えない分を「外から借りる」**ことになっていた
        　（★実測: 外へ **−36,956**。★それが暴走の燃料だった）。
        ★★★実際は ── **原子の大きさは核の質量にほとんど依らない**。
        　★水素もウランも原子半径は同程度。★核は原子の10万分の1。
        ★**核（融合・分裂）と 化学（LJ）は、別のスケール。** ★そこを混ぜていた。
        """
        return np.full((self.n, self.n), SIG0)

    # ── ② にじみ → 重力 ───────────────────────────
    def field(self):
        gi = tuple(np.clip((self.pos[:, k] / L * G).astype(int), 0, G - 1)
                   for k in range(DIM))
        src = np.zeros((G,) * DIM)
        np.add.at(src, gi, self.mass)
        D = 0.9 / (2.0 * DIM)
        for _ in range(RELAX):
            lp = -2.0 * DIM * self.m3
            for ax in range(DIM):
                lp = lp + np.roll(self.m3, 1, ax) + np.roll(self.m3, -1, ax)
            # ★★★源を拡散の係数の中に入れる（2026-09-11 の訂正）。
            #   ★前は外に書いていたので、★**定常状態が「何回回したか」で変わり**、
            #   ★★場が毎歩ふくらみ続けていた。
            self.m3 = self.m3 + D * (lp + src * SRC_K)
            for ax in range(DIM):
                sl = [slice(None)] * DIM
                sl[ax] = 0
                self.m3[tuple(sl)] = 0.0
                sl[ax] = -1
                self.m3[tuple(sl)] = 0.0
        return gi, np.gradient(self.m3)

    # ── ① ポテンシャル ─────────────────────────
    def forces(self):
        d = self.pos[:, None, :] - self.pos[None, :, :]
        r2 = (d * d).sum(-1)
        np.fill_diagonal(r2, np.inf)
        sg = self.sig()
        s2 = sg * sg
        cut = r2 < (RCUT * RCUT) * s2
        # ★★★柔らかい芯（2026-09-11 の訂正）。
        #   ★前は「近すぎたら r を頭打ち」にしていたが、★**力と位置エネルギーが食い違い**、
        #   ★★押し込まれるたびにエネルギーが湧いていた（★実測 +8525）。
        #   ★★★正しくは ── ★芯の中では**力を一定**にして、★位置エネルギーも**直線**にする。
        r = np.sqrt(np.where(np.isfinite(r2), r2, 1.0))
        rmin = np.sqrt(0.70 * s2)
        soft = r < rmin
        rr = np.where(soft, rmin, r)
        inv = (sg / rr) ** 6
        i12 = inv * inv
        ep = self.eps()                                       # ★★種類ごとの強さ
        fmag = 24.0 * ep * (2.0 * i12 - inv) / rr             # ★力の大きさ
        V = 4.0 * ep * (i12 - inv)                            # ★位置エネルギー
        # ★芯の中: ★力は rmin のまま、★位置エネルギーは直線で伸ばす
        V = np.where(soft, V + fmag * (rmin - r), V)
        coef = np.where(cut, fmag / np.maximum(r, 1e-9), 0.0)
        f = (coef[:, :, None] * d).sum(axis=1) / self.mass[:, None]
        self.Elj = float(np.where(cut, V, 0.0).sum()) * 0.5
        return f, r2, cut, s2

    # ── ③ 変換 ── ★★★曲線だけが決める ─────────────
    def transform(self, r2, s2):
        """★融合と分裂。★どちらが起きるかは結合エネルギーの曲線が決める。"""
        n = self.n
        # ★★融合: 十分近く、相対エネルギーが障壁を超え、★**発熱するときだけ**
        near = r2 < (1.35 ** 2) * s2
        ii, jj = np.nonzero(np.triu(near, 1))
        used = np.zeros(n, bool)
        newp, newv, newm, drop, newid = [], [], [], [], []
        gained = 0.0
        for a, b in zip(ii, jj):
            a, b = int(a), int(b)
            if used[a] or used[b]:
                continue
            ma, mb = self.mass[a], self.mass[b]
            mu = ma * mb / (ma + mb)
            dv = self.vel[a] - self.vel[b]
            krel = 0.5 * mu * float(dv @ dv)
            dE = float(binding(np.array([ma + mb]))[0]
                       - binding(np.array([ma]))[0] - binding(np.array([mb]))[0])
            # ★★★2026-09-11 の訂正: ★前は「損をする融合は起きない」と書いていた。
            #   ★★それが間違い。★**払えるエネルギーがあれば、損をする反応も起きる**。
            #   ★実際の宇宙で鉄より重い元素ができたのは、★**超新星の莫大なエネルギー**で
            #     ★吸熱の融合が無理やり起きたから（r過程）。
            #   ★★★僕のルールが1つ消えて、★**エネルギー保存だけになる**。
            # ★★クーロン障壁 ＋ 吸熱なら足りない分
            barrier = COULOMB * (ma * mb) / (ma ** (1.0/3.0) + mb ** (1.0/3.0))
            need = barrier + max(0.0, -dE)
            if krel < need:
                continue
            # ★★★反応が起きる確率（2026-09-11）。★前は「条件を満たせば**必ず**融合」だった。
            #   ★実際の核反応は断面積が小さく、★**ほとんどの衝突では何も起きない**。
            #   ★障壁をどれだけ楽に越えたかで、指数関数的に決まる（★ガモフ因子）。
            #   ★★これが無いと、★重いもの同士が次々に融合して**暴走する**
            #     （★実測: 700個 → 5個、温度1357、吸熱融合が71%）。
            #   ★★★どの衝突が当たるかは、**本物のジッタ**が決める。
            if float(self.jit.take(1)[0]) > math.exp(-barrier / max(1e-9, krel)):
                continue
            m = ma + mb
            p = (self.pos[a] * ma + self.pos[b] * mb) / m
            v = (self.vel[a] * ma + self.vel[b] * mb) / m
            # ★★★生成物は**重心の速度だけ**を持つ（★運動量は保存する）。
            #   ★相対運動 krel と 結合の差 dE を足した分が、★**全部まわりへ出る**。
            #   ★エネルギー保存: krel − B(a) − B(b) = 出る分 − B(m)  →  出る分 = krel + dE
            self.release.append((p.copy(), krel + dE))
            newp.append(p); newv.append(v); newm.append(m)
            newid.append(self.new_pid())
            used[a] = used[b] = True
            drop += [a, b]
            gained += dE
            self.nfuse += 1
            if dE < 0:
                self.nfuse_endo += 1

        # ★★分裂: 重いものは、ときどき割れる。★★**いつ割れるかはジッタが決める**
        heavy = np.nonzero(self.mass > M_PEAK * 1.05)[0]   # ★山を少しでも超えたら候補
        if len(heavy):
            roll = self.jit.take(len(heavy))
            for idx, rr in zip(heavy, roll):
                idx = int(idx)
                if used[idx] or rr > FISSION_P * self.mass[idx]:
                    continue
                m = self.mass[idx]
                # ★★割り方をいくつか試して、★一番得をする割り方を選ぶ。
                #   ★前は1通りだけ試して、損なら諦めていた（★実測: 分裂0回）
                best, bdE = None, 0.0
                for fa in (0.5, 0.42, 0.35, 0.28, 0.2, 0.12):
                    q1, q2 = m * fa, m * (1.0 - fa)
                    if min(q1, q2) < 1.0:
                        continue
                    e = float(binding(np.array([q1]))[0]
                              + binding(np.array([q2]))[0]
                              - binding(np.array([m]))[0])
                    if e > bdE:
                        best, bdE = (q1, q2), e
                if best is None:
                    continue
                m1, m2 = best
                dE = bdE
                u = self.jit.take(DIM) - 0.5
                u /= (np.linalg.norm(u) + 1e-9)
                # ★★破片は親の速度をそのまま（★運動量保存）。★出た分は全部まわりへ
                newp.append(self.pos[idx] + u * 0.6)
                newv.append(self.vel[idx].copy())
                newm.append(m1)
                newid.append(self.new_pid())
                newp.append(self.pos[idx] - u * 0.6)
                newv.append(self.vel[idx].copy())
                newm.append(m2)
                newid.append(self.new_pid())
                self.release.append((self.pos[idx].copy(), dE))
                used[idx] = True
                drop.append(idx)
                gained += dE
                self.nfiss += 1

        if drop:
            keep = np.ones(n, bool)
            keep[np.array(drop)] = False
            self.pos = np.concatenate([self.pos[keep], np.array(newp)])
            self.vel = np.concatenate([self.vel[keep], np.array(newv)])
            self.mass = np.concatenate([self.mass[keep], np.array(newm)])
            self.pid = np.concatenate([self.pid[keep],
                                       np.array(newid, dtype=np.int64)])
            assert len(self.pid) == len(self.mass), '名札の数が合わない'
        self.heat = gained
        return gained

    def reach(self):
        """★光が届く距離。★塊の広がりに合わせる。"""
        r = np.sqrt(((self.pos - self.c) ** 2).sum(1))
        self.reach2 = max(36.0, (2.5 * float(r.mean()) + 4.0) ** 2)

    def deliver(self):
        """★★★出た熱を、まわりへ配る（★粗い放射輸送）。
        ★実際の光は出した本人から離れて、★**別の場所で吸われる**。
        ★★これが無いと、融合した本人だけが速くなって暴走する。"""
        if not self.release:
            return 0.0
        tot = 0.0
        area = np.full(self.n, np.pi * (0.5 * SIG0) ** 2)   # ★粒子の断面積
        for p, e in self.release:
            d = self.pos - p
            r2 = (d * d).sum(1) + 0.25
            # ★★★吸収率（2026-09-11）。★前は「届く範囲に必ず全部配る」だった。
            #   ★実際の光は**当たった分だけ吸われて、残りは宇宙へ抜ける**。
            #   ★粒子が数個になると、★★**ほとんど抜ける**（★だから冷える）。
            #   ★w = 断面積 ÷ 4πr²  ＝ その粒子が受け止める割合
            w = area / (4.0 * np.pi * r2)
            ws = float(w.sum())
            if not np.isfinite(ws) or ws <= 0:
                self.Eout += e
                tot += e
                continue
            hit = min(1.0, ws)                       # ★吸われる割合
            share = e * (w / ws) * hit
            u = d / (np.sqrt((d * d).sum(1))[:, None] + 1e-9)
            vu = (self.vel * u).sum(1)
            s_ = -vu + np.sqrt(np.maximum(0.0, vu * vu + 2.0 * share / self.mass))
            k_before = self._ke()
            self.vel += u * s_[:, None]
            got = self._ke() - k_before
            # ★★吸われなかった分＋配りきれなかった分は、★そのまま宇宙へ
            self.Eout += (e - got)
            tot += e
        self.release = []
        return tot

    # ── ★★★エネルギーの台帳（★変換が本体）─────────────
    def ledger(self):
        """★どのエネルギーが、いまどこに在るか。★合計が保存していれば実装が正しい。"""
        Ebind = float(binding(self.mass).sum())      # ★質量の中に残っている燃料
        Ekin = 0.5 * float((self.mass * (self.vel * self.vel).sum(1)).sum())
        return Ebind, Ekin

    def Wgrav_step(self):
        w = self.Wgrav - self._wprev
        self._wprev = self.Wgrav
        return w

    def _ke(self):
        return 0.5 * float((self.mass * (self.vel * self.vel).sum(1)).sum())

    def step(self):
        """★★★どの操作でエネルギーが湧いているかを、1歩ごとに突き合わせる。"""
        self.step_n += 1
        f, r2, cut, s2 = self.forces()
        lj0 = self.Elj
        gi, grads = self.field()
        for ax in range(DIM):
            f[:, ax] += GRAV * grads[ax][gi]

        # ── ★★★① 動き ── velocity Verlet（シンプレクティック法）
        #   ★前はオイラー法で、★**エネルギーが湧いていた**（★実測: +2684）。
        #   ★★レナード・ジョーンズのような急峻なポテンシャルでは教科書どおりの症状。
        #   ★分子動力学は必ずこの形を使う。
        k0 = self._ke()
        self.vel = self.vel + 0.5 * f * DT          # ★半分だけ蹴る
        x0 = self.pos.copy()
        self.pos = self.pos + self.vel * DT         # ★動かす
        dx = self.pos - x0
        f1, r2, cut, s2 = self.forces()             # ★新しい場所で力を測り直す
        # ★★場は1歩に1回だけ（★前は2回呼んでいて、そのぶん余計にふくらんでいた）
        gi1 = tuple(np.clip((self.pos[:, k] / L * G).astype(int), 0, G - 1)
                    for k in range(DIM))
        gacc = np.zeros_like(f1)
        for ax in range(DIM):
            gacc[:, ax] = GRAV * grads[ax][gi1]
        # ★★★重力がした仕事を数える（2026-09-11）。
        #   ★前はこれを台帳に入れておらず、★★**湧いているように見えていた**（実測 +2991）。
        #   ★実際は「潰れると熱くなる」という**本物の物理**だった。
        gold = np.zeros_like(f)
        for ax in range(DIM):
            gold[:, ax] = GRAV * grads[ax][gi]
        self.Wgrav += float((self.mass[:, None] * 0.5 * (gold + gacc) * dx).sum())
        f1 = f1 + gacc
        self.vel = self.vel + 0.5 * f1 * DT         # ★残り半分を蹴る
        k1 = self._ke()
        self.d_move += (k1 - k0) + (self.Elj - lj0) - self.Wgrav_step()

        # ── ★② 変換 ─────────────────────────
        k0 = self._ke()
        b0 = float(binding(self.mass).sum())
        lj_before = self.Elj                      # ★★変換の前の LJ 位置エネルギー
        self.transform(r2, s2)
        self.forces()                             # ★★σ が変わったので測り直す
        lj_after = self.Elj
        k1 = self._ke()
        b1 = float(binding(self.mass).sum())
        # ★★★σ は質量で変わる（σ ∝ m^⅓）。★融合すると**同じ距離でも LJ が飛ぶ**。
        #   ★前はその飛びを帳簿に入れておらず、★**それが最後の湧き口だった**（実測 +19862）。
        #   → ★飛んだぶんを、★**出る熱から差し引く**（★エネルギーはそこから来ている）。
        dlj = lj_after - lj_before
        # ★★★スケールで掛けたら**符号が反転して爆発した**（★実測: 700個→1個、外へ48万）。
        #   ★素直に足し引きする ── ★LJ が上がったぶんは、★**出る熱から引く**。
        #   ★引いた結果が負になるなら、★**その分は世界のどこかから借りた**ことにして記録する。
        # ★★σ が質量に依らなくなったので、★飛びはほぼゼロになるはず。
        #   ★それでも残る分は、★**出る熱から引くだけ**（★借金はしない）。
        if abs(dlj) > 1e-12:
            tot0 = sum(e for _p, e in self.release)
            if tot0 >= dlj:
                sc = (tot0 - dlj) / tot0 if tot0 > 1e-12 else 0.0
                self.release = [(p, e * sc) for p, e in self.release]
            else:
                self.release = []
                self.Eout -= (dlj - tot0)      # ★（★ここに来なくなるはず）
        want = sum(e for _p, e in self.release)
        # ★消えた運動 ＋ 増えた結合 − LJ の飛び ＝ これから配る分 であるべき
        self.d_tr += (k0 - k1) + (b1 - b0) - dlj - want

        # ── ★③ 配る ─────────────────────────
        k0 = self._ke()
        self.reach()
        gave = self.deliver()
        k1 = self._ke()
        self.d_del += (k1 - k0) - gave             # ★★0であるべき
        self.Ein += gave

        # ── ★④ 放射 ─────────────────────────
        rad = self.pos - self.c
        r = np.sqrt((rad * rad).sum(1)) + 1e-9
        T = float((self.mass * (self.vel * self.vel).sum(1)).sum()) / (DIM * self.n)
        outer = 1.0 + 6.0 * (r / R_MAX) ** 2
        damp = np.clip(1.0 - RAD0 * (1.0 + (T / T_REF) ** 1.5) * outer, 0.85, 0.9999)
        ke0 = self._ke()
        self.vel *= damp[:, None]
        self.Eout += (ke0 - self._ke())

        keep = np.sqrt(((self.pos - self.c) ** 2).sum(1)) < R_MAX
        if not keep.all():
            self.escaped += int((~keep).sum())
            self.mass_esc += float(self.mass[~keep].sum())
            self.Eout += 0.5 * float((self.mass[~keep]
                                      * (self.vel[~keep] ** 2).sum(1)).sum())
            self.Eout -= float(binding(self.mass[~keep]).sum())
            self.pos, self.vel, self.mass = (self.pos[keep], self.vel[keep],
                                             self.mass[keep])
            self.pid = self.pid[keep]
        self._r = r
        return T

    def molecules(self):
        """★★★分子 ── **LJ で束縛された塊**。★核種の上の階層。

        ★束縛の定義は物理のまま: ★**対の全エネルギーが負**。★閾値はゼロ。
        ★★僕は「分子とは何か」を書いていない。
        """
        n = self.n
        d = self.pos[:, None, :] - self.pos[None, :, :]
        r2 = (d * d).sum(-1)
        np.fill_diagonal(r2, np.inf)
        sg = self.sig()
        s2 = sg * sg
        cut = r2 < (RCUT * RCUT) * s2
        r = np.sqrt(np.where(np.isfinite(r2), r2, 1.0))
        rmin = np.sqrt(0.70 * s2)
        rr = np.where(r < rmin, rmin, r)
        inv = (sg / rr) ** 6
        ep = self.eps()
        V = 4.0 * ep * (inv * inv - inv)
        dv = self.vel[:, None, :] - self.vel[None, :, :]
        mu = (self.mass[:, None] * self.mass[None, :]) / (
            self.mass[:, None] + self.mass[None, :])
        bound = cut & ((V + 0.5 * mu * (dv * dv).sum(-1)) < 0.0)
        ii, jj = np.nonzero(np.triu(bound, 1))
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
        grp = {}
        for i in range(n):
            grp.setdefault(find(i), []).append(i)
        sizes, comp, group = [], {}, []
        for _root, mem in grp.items():
            if len(mem) < 2:
                continue
            sizes.append(len(mem))
            # ★★中身そのもの（★質量を並べたもの）。★これが「分子式」にあたる
            f = tuple(sorted(int(round(float(self.mass[i]))) for i in mem))
            comp[f] = comp.get(f, 0) + 1
            # ★★★名札の組。★これが同じなら「同じ分子」
            group.append((f, frozenset(int(self.pid[i]) for i in mem)))
        sizes.sort(reverse=True)
        return sizes, len(comp), len(ii), comp, group

    def stats(self, T):
        m = self.mass
        r = np.sqrt(((self.pos - self.c) ** 2).sum(1))
        v2 = (self.vel * self.vel).sum(1)

        # ── ★★★エネルギーの台帳（★変換が本体）────────────
        Ebind = float(binding(m).sum())              # ★質量の中に残っている燃料
        Ekin = 0.5 * float((m * v2).sum())           # ★運動
        # ★★★台帳の全部（2026-09-11 に2度直した）:
        #   ★結合エネルギーは**負の位置エネルギー**（符号）
        #   ★★**LJ の位置エネルギー**と**重力がした仕事**が抜けていた
        Etot = Ekin + self.Elj + self.Eout - Ebind - self.Wgrav
        if self.E0tot is None:
            self.E0tot = Etot
        # ★★★ずれの割り方（2026-09-11 の訂正）。
        #   ★前は**初期値**で割っていた。★でも初期値は −312 しかなく（★全部が質量1）、
        #   ★★**ゼロに近い数で割っていた**ので、★絶対値 +318 が「+102%」に見えていた。
        #   ★★★正しくは**流れているエネルギーの大きさ**で割る。
        scale = abs(Ekin) + abs(self.Elj) + abs(self.Eout) + abs(Ebind) + abs(self.Wgrav)
        drift = (Etot - self.E0tot) / max(1.0, scale)
        drift_abs = Etot - self.E0tot

        # ── ★質量の台帳 ────────────────────────
        mnow = float(m.sum())
        mdrift = (mnow + self.mass_esc - self.mass0) / max(1e-9, self.mass0)

        # ── ★種類 ─────────────────────────────
        mr = np.round(m, 2)
        kinds = len(np.unique(mr))
        _, cts = np.unique(mr, return_counts=True)
        pk = cts / cts.sum()
        shannon = float(-(pk * np.log(pk)).sum())     # ★★種類の多様性

        # ── ★階層（★質量のスケール）─────────────────
        lb = np.log2(np.maximum(m, 1.0))
        hist, _ = np.histogram(lb, bins=np.arange(0, max(1.5, lb.max() + 1.5), 1.0))
        mscales = int((hist > max(2, 0.03 * len(m))).sum())

        # ── ★階層（★空間の層）─────────────────────
        edges = np.percentile(r, np.linspace(0, 100, 11))
        prof, tprof = [], []
        for k in range(10):
            sel = (r >= edges[k]) & (r <= edges[k + 1])
            prof.append(float(m[sel].mean()) if sel.sum() > 3 else 0.0)
            tprof.append(float(v2[sel].mean()) / DIM if sel.sum() > 3 else 0.0)
        prof, tprof = np.array(prof), np.array(tprof)
        rng = prof.max() - prof.min()
        layers = 1 + (int((np.abs(np.diff(prof)) > 0.25 * rng).sum())
                      if rng > 1e-6 else 0)

        # ── ★流れ ─────────────────────────────
        q1, q3 = np.percentile(r, 25), np.percentile(r, 75)
        Tin = float(v2[r < q1].mean()) / DIM if (r < q1).any() else 0.0
        Tout = float(v2[r > q3].mean()) / DIM if (r > q3).any() else 0.0

        # ── ★★周期に入っていないか ──────────────────
        # ★★★周期の測り方を直した（2026-09-11）。
        #   ★前は粗いマクロ状態1個を見ていて、★**取りうる値が約7,000通りしかなく**、
        #   ★★37,700歩も回せば**鳩の巣原理で必ずぶつかった**（★実測 再訪688）。
        #   ★★★周期とは「同じ状態に戻って、**その後も同じ順番で辿る**」こと。
        #   → ★**直近8歩の並び**が前にも出たかを見る。★偶然では起きない。
        macro = (kinds, round(float(m.max()), 1), round(T, 2),
                 round(float(r.mean()), 2), self.n, layers,
                 round(shannon, 3), mscales)
        self.trail.append(hash(macro))
        del self.trail[:-8]
        if len(self.trail) == 8:
            hs = hash(tuple(self.trail))
            if hs in self.seen and self.step_n - self.seen[hs] > 40:
                self.revisits += 1
            self.seen[hs] = self.step_n
        newness = len(self.seen) / max(1, self.step_n / 5.0)   # ★新しい並びの割合

        msz, mkinds, nb, comp, group = self.molecules()

        # ── ★★★図鑑 ── 何ができたか、そのまま並べる ────────────
        #   ★原子の名前は付けない。★**質量そのものが種類**（★人間の周期表は書いていない）。
        #   ★反応するかどうかは、★結合エネルギーの曲線から出てくる（★書いていない）。
        uu, cc = np.unique(np.round(m).astype(int), return_counts=True)
        rr = reactivity(uu.astype(float))
        bb = binding(uu.astype(float)) / np.maximum(1.0, uu.astype(float))
        now_a = {int(uu[k]): int(cc[k]) for k in range(len(uu))}
        for k in range(len(uu)):
            self.zukan_a.setdefault(int(uu[k]), [self.step_n, 0])[1] += 1
        # ★★★分子の寿命 ── **同じ名札の組**が続けて束縛されている歩数。
        #   ★2026-09-11 の訂正: 前は「同じ中身」で数えていて、
        #   ★★よく出る組み合わせはカウントが繋がり、★嘘の記録が出ていた。
        run = {}
        for f, pset in group:
            z = self.zukan_m.setdefault(f, [self.step_n, 0, 0])
            while len(z) < 3:
                z.append(0)
            run[pset] = self.mol_run.get(pset, 0) + 5     # ★stats は5歩ごと
            if run[pset] > z[2]:
                z[2] = run[pset]
        for f in comp:
            self.zukan_m[f][1] += 1
        self.mol_run = run
        mol_life = max(list(run.values()) + [0])          # ★いま生きている古株
        mol_best = max([v[2] for v in self.zukan_m.values()] or [0])

        atoms = []
        for mm_ in sorted(self.zukan_a):
            r_ = float(reactivity(np.array([float(mm_)]))[0])
            atoms.append(dict(m=mm_, name=sym(mm_), n=now_a.get(mm_, 0),
                              react=round(r_, 2), first=self.zukan_a[mm_][0]))
        # ★★★並べ方（2026-09-11 の訂正）。
        #   ★前は「大きい順」だけだった。★そのせいで**一番長生きした分子が出ない**
        #     （★実測: 最長1160歩と出ているのに、並んでいる分子は全部0歩）。
        #   ★★長生き順 → 大きい順 の順に混ぜる。
        _life = lambda kv: (kv[1][2] if len(kv[1]) > 2 else 0)
        by_life = sorted(self.zukan_m.items(), key=lambda kv: -_life(kv))[:20]
        by_size = sorted(self.zukan_m.items(),
                         key=lambda kv: (-len(kv[0]), -kv[1][1]))[:20]
        mk, _got = [], set()
        for kv in by_life + by_size:
            if kv[0] in _got:
                continue
            _got.add(kv[0])
            mk.append(kv)
        mols = [dict(f=formula(f), size=len(f), n=int(comp.get(f, 0)),
                     mass=int(sum(f)), times=int(v[1]), first=int(v[0]),
                     life=int(v[2] if len(v) > 2 else 0)) for f, v in mk]
        jh, ju = self.jit.health()
        return dict(
            atoms=atoms, mols=mols, mollife=mol_life, molbest=mol_best,
            step=self.step_n, n=self.n, T=round(T, 3),
            mol=len(msz), molmax=(msz[0] if msz else 0), molkinds=mkinds,
            bonds=nb,
            # ★★★エネルギー
            Ebind=round(Ebind, 1), Ekin=round(Ekin, 1), Eout=round(self.Eout, 1),
            Ein=round(self.Ein, 1), drift=round(drift * 100, 3),
            dabs=round(drift_abs, 1),
            fuel=round(100.0 * Ebind / max(1e-9, self.Ebind_max), 1),
            # ★質量
            mass=round(mnow, 1), mesc=round(self.mass_esc, 1),
            mdrift=round(mdrift * 100, 4),
            # ★種類
            kinds=kinds, shannon=round(shannon, 3),
            mmax=round(float(m.max()), 1), mmean=round(float(m.mean()), 2),
            mmed=round(float(np.median(m)), 2),
            # ★階層
            mscales=mscales, layers=layers,
            # ★★numpy の数は JSON にできない。★float() に落とす（2026-09-11 のバグ）
            prof=[float(round(float(x), 2)) for x in prof],
            tprof=[float(round(float(x), 3)) for x in tprof],
            # ★流れ
            Tin=round(Tin, 3), Tout=round(Tout, 3),
            grad=round(Tin / max(1e-9, Tout), 2), R=round(float(r.mean()), 2),
            # ★反応
            fuse=self.nfuse, fiss=self.nfiss, endo=self.nfuse_endo,
            heat=round(self.heat, 2),
            # ★周期・揺さぶり
            revisit=self.revisits, newness=round(newness, 3),
            esc=self.escaped, jit=round(jh, 3), juniq=ju,
            # ★★★どの操作で湧いたか
            dmove=round(self.d_move, 1), dtr=round(self.d_tr, 1),
            ddel=round(self.d_del, 1), Wg=round(self.Wgrav, 1),
            Elj=round(self.Elj, 1))

    # ── ★★★続きのための紙 ────────────────────────
    #   ★保存するのは**世界そのもの**（位置・速さ・質量・にじみの場）と
    #   ★**台帳**（湧いた分・逃げた分・外へ出た分）と**指紋**（再訪の記憶）。
    #   ★ジッタは保存しない ── ★崩壊の「いつ」は毎回あたらしく実行時間から取る。
    SCALARS = ("step_n", "secs", "revisits", "escaped", "nfuse", "nfiss", "nfuse_endo",
               "heat", "Ein", "Eout", "mass0", "Ebind_max", "mass_esc",
               "E0tot", "d_move", "d_tr", "d_del", "Wgrav", "_wprev",
               "reach2", "Elj")

    def save(self, path, hist=None):
        """★★途中で電源が切れても壊れないように、★別名で書いてから置き換える"""
        v = []
        for k in self.SCALARS:
            x = getattr(self, k)
            v.append(np.nan if x is None else float(x))
        np.savez_compressed(
            path + ".tmp.npz",
            pos=self.pos, vel=self.vel, mass=self.mass, m3=self.m3,
            pid=self.pid, next_pid=np.array([self.next_pid]),
            trail=np.array(self.trail, dtype=np.int64),
            seen_k=np.array(list(self.seen.keys()), dtype=np.int64),
            seen_v=np.array(list(self.seen.values()), dtype=np.int64),
            scalars=np.array(v, dtype=np.float64),
            names=np.array(self.SCALARS),
            # ★★★図鑑も持ち越す。★これが無いと、続きのたびに**出来たものを忘れる**
            zk_a=np.array(["%d:%d:%d" % (k, w[0], w[1])
                           for k, w in self.zukan_a.items()]),
            zk_m=np.array(["%s:%d:%d:%d" % ("+".join(str(x) for x in k),
                                            w[0], w[1], w[2] if len(w) > 2 else 0)
                           for k, w in self.zukan_m.items()]))
        os.replace(path + ".tmp.npz", path)
        # ★★★画面が /state.json から読むのと**まったく同じ形**で置く。
        #   ★こうすると、★世界が止まっていても、★同じ画面がそのまま開ける。
        try:
            d = json.loads(live.snap())
            d["running"] = False
            d["step"] = self.step_n
            d["saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with io.open(HIST_PATH + ".tmp", "w", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False))
            os.replace(HIST_PATH + ".tmp", HIST_PATH)
        except Exception as e:
            print("★★紙が置けなかった: %r" % (e,), flush=True)

    def load(self, path, hist=None):
        z = np.load(path, allow_pickle=False)
        self.pos, self.vel = z["pos"], z["vel"]
        self.mass, self.m3 = z["mass"], z["m3"]
        if "pid" in z.files:
            self.pid = z["pid"]
            self.next_pid = int(z["next_pid"][0])
        else:                                  # ★名札の無い古い紙
            self.pid = np.arange(len(self.mass), dtype=np.int64)
            self.next_pid = len(self.mass)
        self.trail = [int(x) for x in z["trail"]]
        self.seen = {int(k): int(v) for k, v in zip(z["seen_k"], z["seen_v"])}
        # ★★名前で戻す。★項目が増えても古い紙が読める
        names = [str(x) for x in z["names"]]
        for k, x in zip(names, z["scalars"]):
            if k in self.SCALARS:
                setattr(self, k, None if np.isnan(x) else
                        (int(x) if k in ("step_n", "revisits", "escaped",
                                         "nfuse", "nfiss", "nfuse_endo")
                         else float(x)))
        self.release = []
        # ★★★図鑑も戻す。★これが無いと、続きのたびに**出来たものを忘れる**
        self.zukan_a, self.zukan_m = {}, {}
        for t in (z["zk_a"] if "zk_a" in z.files else []):
            a, b, c = str(t).split(":")
            self.zukan_a[int(a)] = [int(b), int(c)]
        self.mol_run = {}
        for t in (z["zk_m"] if "zk_m" in z.files else []):
            q = str(t).split(":")
            self.zukan_m[tuple(int(x) for x in q[0].split("+"))] = [
                int(q[1]), int(q[2]), int(q[3]) if len(q) > 3 else 0]
        # ★★グラフの過去も戻す。★これが無いと、続きのたびに**線が消える**
        if hist is not None and os.path.exists(HIST_PATH):
            try:
                old = json.load(io.open(HIST_PATH, encoding="utf-8")).get("hist", {})
                for k in hist:
                    hist[k][:] = list(old.get(k, []))[-500:]
            except Exception:
                pass


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    live.start()
    jit = Jitter()
    w = Earth(jit)
    hist = {k: [] for k in HIST_KEYS}
    if RESUME and os.path.exists(STATE_PATH):
        try:
            w.load(STATE_PATH, hist)
            print("★%d歩目の続きから始める（粒%d 質量%.0f）"
                  % (w.step_n, w.n, w.mass.sum()), flush=True)
        except Exception as e:
            print("★★続きが読めなかった（%r）。★最初から始める" % (e,), flush=True)
    live.put(running=True, cfg=dict(N=w.n, L=L, dim=DIM, coulomb=COULOMB))
    live.log("★地球 v2 ── ★★法則4つ。★熱源は世界の中（変換）だけ")
    live.log("★★★最下層に置いたのは**結合エネルギーの曲線1本**。★元素の一覧は書いていない")
    live.log("★崩壊がいつ起きるかは、★★本物の実行時間のジッタが決める")
    t0 = time.time()
    base_secs = w.secs                    # ★★前の区切りまでの合計
    stop_at = (w.step_n + MAX_STEPS) if MAX_STEPS > 0 else 0
    while True:
        T = w.step()
        w.secs = base_secs + (time.time() - t0)
        if w.step_n % 5 == 0:
            st = w.stats(T)
            for k in hist:
                hist[k].append(st[k])
                del hist[k][:-500]
            live.put(stat=dict(st, min=round(w.secs / 60.0, 1)),
                     hist=hist, L=L,
                     pos=[[round(float(p[0]), 2), round(float(p[1]), 2),
                           round(float(p[2]), 2)] for p in w.pos],
                     deg=[float(round(x, 2)) for x in w.mass])
        if w.step_n % 100 == 0:
            st = w.stats(T)
            live.log("%5d歩 T%.3f 燃料%.0f%% ★種類%d 多様%.2f ★層%d/質量段%d "
                     "融合%d 分裂%d 再訪%d 新しさ%.2f"
                     % (st["step"], st["T"], st["fuel"], st["kinds"], st["shannon"],
                        st["layers"], st["mscales"], st["fuse"], st["fiss"],
                        st["revisit"], st["newness"]))
            print("%5d T%.3f 中%.2f/外%.2f 勾配%.1f | 燃料%5.1f%% 運動%7.0f 外へ%8.0f "
                  "ずれ%+.3f%%(絶対%+.0f) | 質量%6.0f 逃%5.0f ずれ%+.4f%% | ★種類%3d 多様%.2f "
                  "最大%6.1f 中央%5.1f | ★層%d 質量段%d | 融合%4d(吸%3d) 分裂%4d | "
                  "粒%4d 逃%3d 再訪%2d 新しさ%.2f | ★分子%3d(最大%3d 種類%3d 結合%4d いま最古%4d歩 記録%5d歩)"
                  % (st["step"], st["T"], st["Tin"], st["Tout"], st["grad"],
                     st["fuel"], st["Ekin"], st["Eout"], st["drift"], st["dabs"],
                     st["mass"], st["mesc"], st["mdrift"], st["kinds"],
                     st["shannon"], st["mmax"], st["mmed"], st["layers"],
                     st["mscales"], st["fuse"], st["endo"], st["fiss"],
                     st["n"], st["esc"], st["revisit"], st["newness"],
                     st["mol"], st["molmax"], st["molkinds"], st["bonds"],
                     st["mollife"], st["molbest"]),
                  flush=True)
            print("      ★★湧いた内訳: 動き%+9.1f 変換%+7.1f 配る%+9.1f | "
                  "重力の仕事%+10.1f  LJ位置%+9.1f"
                  % (st["dmove"], st["dtr"], st["ddel"], st["Wg"], st["Elj"]),
                  flush=True)
        if SAVE_EVERY and w.step_n % SAVE_EVERY == 0:
            w.save(STATE_PATH, hist)
        if MAX_MIN and (time.time() - t0) >= MAX_MIN * 60.0:
            w.save(STATE_PATH, hist)
            print("★%.1f分で一区切り（%d歩目）。★状態を置いた"
                  % ((time.time() - t0) / 60.0, w.step_n), flush=True)
            return
        if stop_at and w.step_n >= stop_at:
            w.save(STATE_PATH, hist)
            print("★%d歩で一区切り。★状態を置いた（%s）"
                  % (w.step_n, os.path.basename(STATE_PATH)), flush=True)
            return


if __name__ == "__main__":
    main()
