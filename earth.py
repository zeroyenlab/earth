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


# ── ★★★電荷（2026-09-11）──────────────────────
#   ★粒が持つ2つ目の量。★符号がある。★★法則は1つも増えない。
#   ★曲線が B(m) から **B(m,q) の谷** になるだけ。
#   ★★これは本物の「安定の谷」そのもの（★核図表の Z 対 N）。
#   ★谷から外れたかたまりは、★電荷を含む割り方で分裂した方が得になる
#     → ★★**軽くて電荷を持つかけらが出る**。★それが β崩壊であり、出るものが電子。
#   ★EARTH_CHARGE=0 のときは、★今までと**1ビットも変わらない**。
Q_ON = os.environ.get("EARTH_CHARGE", "0") == "1"
Z_FRAC = float(os.environ.get("EARTH_ZFRAC", 0.5))   # ★谷の位置（★q ≒ 0.5·m）
S_Q = float(os.environ.get("EARTH_SQ", 0.8))         # ★谷の広さ
# ★★かけらの質量の下限。★ゼロにすると割り算が壊れるので置くだけの数。
#   ★★「電子の質量」ではない ── ★どれだけ軽いものが出るかは曲線が決める。
MFRAG_MIN = float(os.environ.get("EARTH_MFRAG", 1e-4))
# ★★電気の強さ。★重力の GRAV と同じ立場の数（★力の単位を決めるだけ）。
KQ = float(os.environ.get("EARTH_KQ", 60.0))


def binding(m, q=None):
    """★★★結合エネルギー。★これ1本で融合も分裂も決まる。

    ★B(m)/m が M_PEAK で最大になる山。
    ★★軽い同士の融合は発熱、★重いものの分裂は発熱、★その逆は起きない。
    ★★★僕は「何が安定か」を書いていない。★曲線が決める。

    ★★★q を渡すと、★山は**谷**になる（2026-09-11）。
    　★谷底は q = Z_FRAC·m。★そこから離れるほど結合が浅くなる。
    　★広がりを √m に比例させるのは、★重いものほど許容が広いから（★本物もそう）。
    """
    x = np.log(np.maximum(m, 1e-9) / M_PEAK)
    b = E0 * m * np.exp(-(x * x) / (2.0 * S_W * S_W))
    if q is None or not Q_ON:
        return b
    dq = q - Z_FRAC * m
    return b * np.exp(-(dq * dq) / (2.0 * S_Q * S_Q * np.maximum(m, 1.0)))


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


def reactivity(m, q=None):
    """★0 = 谷の底（不活性） 〜 1 = 谷から遠い（反応する）"""
    return 1.0 - binding(m, q) / np.maximum(1e-9, E0 * m)
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
        # ★★★電荷。★★全部ゼロから始まる（★中性子だけの世界）。
        #   ★谷底は q = 0.5·m なので、★質量1・電荷0 は**谷から外れている**。
        #   → ★電荷を吐いた方が得になる。★★僕は電子を1つも置いていない。
        self.q = np.zeros(n)
        self.q0 = 0.0
        self.q_esc = 0.0
        # ★★★名札。★「同じ分子が壊れずに続いているか」を測るのに要る。
        #   ★中身（質量の並び）だけでは、★**別の粒で出来た別の分子**と区別できない。
        self.pid = np.arange(n, dtype=np.int64)
        self.next_pid = int(n)
        self.m3 = np.zeros((G,) * DIM)
        # ★★★電荷のにじみ。★質量のにじみと**まったく同じ法則**をもう1枚に当てるだけ
        self.q3 = np.zeros((G,) * DIM)
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
        self.q0 = float(self.q.sum())
        # ★★燃料の満タン ＝ 全部が山の質量になったときの結合エネルギー
        self.Ebind_max = float(binding(np.array([M_PEAK]))[0]) * self.mass0 / M_PEAK
        self.mass_esc = 0.0
        self.E0tot = None
        # ★★★診断: どの操作でいくら湧いたか
        self.d_move = self.d_tr = self.d_del = 0.0
        self.Wgrav = self._wprev = 0.0
        self.Wq = self._wqprev = 0.0
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
        """★★結びつきの強さ。★★★質量から決まる。★僕は表を書かない。

        ★★★2026-09-11: 電荷があるときは**谷からの外れ方**も効く。
        　★同じ曲線 B(m,q) を使うだけ。★新しい表は書かない。
        　★谷から外れている＝結合が浅い＝反応する。★★イオンが反応するのと同じ向き。
        """
        e = EPS_MIN + (EPS_MAX - EPS_MIN) * reactivity(self.mass,
                                                       self.q if Q_ON else None)
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
        srcq = None
        if Q_ON:
            srcq = np.zeros((G,) * DIM)
            np.add.at(srcq, gi, self.q)          # ★★源が電荷になるだけ
        D = 0.9 / (2.0 * DIM)
        for _ in range(RELAX):
            if Q_ON:
                lq = -2.0 * DIM * self.q3
                for ax in range(DIM):
                    lq = lq + np.roll(self.q3, 1, ax) + np.roll(self.q3, -1, ax)
                self.q3 = self.q3 + D * (lq + srcq * SRC_K)
                for ax in range(DIM):
                    sl = [slice(None)] * DIM
                    sl[ax] = 0
                    self.q3[tuple(sl)] = 0.0
                    sl[ax] = -1
                    self.q3[tuple(sl)] = 0.0
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
        return gi, np.gradient(self.m3), (np.gradient(self.q3) if Q_ON else None)

    # ── ① ポテンシャル ─────────────────────────
    def forces(self):
        d = self.pos[:, None, :] - self.pos[None, :, :]
        r2 = (d * d).sum(-1)
        np.fill_diagonal(r2, np.inf)
        sg = self.sig()
        s2 = sg * sg
        cut = r2 < (RCUT * RCUT) * s2
        # ★★★原子でないものは、化学に参加しない（2026-09-11）。
        #   ★この世界の物質の単位は**1**（★全部そこから始まった）。
        #   ★それより軽いかけらは原子ではない。★LJ は原子どうしの法則。
        #   ★★前に σ∝m^⅓ で「核のスケールと化学のスケールを混ぜた」のと同じ間違い。
        #   ★実測: 混ぜたままだと、★質量0.014 の加速度が70倍になり
        #     ★**velocity Verlet が保たない**（★動き −100.8 / 温度 外830）。
        if Q_ON:
            _atom = self.mass >= 1.0
            cut = cut & _atom[:, None] & _atom[None, :]
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
        newq = []
        gained = 0.0
        for a, b in zip(ii, jj):
            a, b = int(a), int(b)
            if used[a] or used[b]:
                continue
            ma, mb = self.mass[a], self.mass[b]
            mu = ma * mb / (ma + mb)
            dv = self.vel[a] - self.vel[b]
            krel = 0.5 * mu * float(dv @ dv)
            qa, qb = self.q[a], self.q[b]
            dE = float(binding(np.array([ma + mb]), np.array([qa + qb]))[0]
                       - binding(np.array([ma]), np.array([qa]))[0]
                       - binding(np.array([mb]), np.array([qb]))[0])
            # ★★★2026-09-11 の訂正: ★前は「損をする融合は起きない」と書いていた。
            #   ★★それが間違い。★**払えるエネルギーがあれば、損をする反応も起きる**。
            #   ★実際の宇宙で鉄より重い元素ができたのは、★**超新星の莫大なエネルギー**で
            #     ★吸熱の融合が無理やり起きたから（r過程）。
            #   ★★★僕のルールが1つ消えて、★**エネルギー保存だけになる**。
            # ★★クーロン障壁 ＋ 吸熱なら足りない分
            # ★★★2026-09-11: 電荷が世界に在るので、★**代用をやめて本物を使う**。
            #   ★これまでは電荷が無かったので質量で代用していた（★コメントにそう書いてある）。
            #   ★★中性どうしには障壁が無い ── ★これは本物の中性子捕獲と同じ。
            #   ★逆符号なら障壁は負（★引き合う）。★そこは0で止める。
            if Q_ON:
                barrier = COULOMB * max(0.0, float(qa * qb)) / (
                    ma ** (1.0/3.0) + mb ** (1.0/3.0))
            else:
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
            newq.append(qa + qb)                 # ★★電荷は足されるだけ（★保存）
            newid.append(self.new_pid())
            used[a] = used[b] = True
            drop += [a, b]
            gained += dE
            self.nfuse += 1
            if dE < 0:
                self.nfuse_endo += 1

        # ★★分裂: 重いものは、ときどき割れる。★★**いつ割れるかはジッタが決める**
        if Q_ON:
            # ★★★谷から外れているものは、★重さに関係なく候補（2026-09-11）。
            #   ★得かどうかは曲線が決める。★僕は「どれが崩壊するか」を書かない。
            heavy = np.nonzero((self.mass > M_PEAK * 1.05)
                               | (np.abs(self.q - Z_FRAC * self.mass) > 1e-9))[0]
        else:
            heavy = np.nonzero(self.mass > M_PEAK * 1.05)[0]  # ★山を超えたら候補
        if len(heavy):
            roll = self.jit.take(len(heavy))
            for idx, rr in zip(heavy, roll):
                idx = int(idx)
                if used[idx] or rr > FISSION_P * self.mass[idx]:
                    continue
                m = self.mass[idx]
                Qp = float(self.q[idx])
                # ★★割り方をいくつか試して、★一番得をする割り方を選ぶ。
                #   ★前は1通りだけ試して、損なら諦めていた（★実測: 分裂0回）
                # ★★★電荷を入れると、★**とても軽いかけら**も候補になる（2026-09-11）。
                #   ★質量の下限1.0 は「核どうしの分裂」を想定した値だった。
                #   ★★電荷だけを持ち出すかけらは、★それよりずっと軽くていい。
                #   ★どれだけ軽いかは**曲線が決める**。★僕は電子の質量を書かない。
                FA = ((0.5, 0.42, 0.35, 0.28, 0.2, 0.12)
                      + ((0.02, 0.004, 0.001) if Q_ON else ()))
                # ★★★電荷は整数でしか動かない（2026-09-11）。
                #   ★実際の物理でも電荷の量子化は**経験的な入力**であって、
                #   ★量子力学から出るものではない（★角運動量の量子化とは別の話）。
                #   ★★前は 0.25 や 0.5 も候補にしていたので電荷が連続になり、
                #   ★**元素の境目がぼやけていた**（★整数に寄るのは 42% だけだった）。
                DQ = (0.0,) if not Q_ON else (-2.0, -1.0, 0.0, 1.0, 2.0)
                # ★★候補を全部いっぺんに評価する（★81通りを1回の numpy で）
                fa = np.array(FA)[:, None]
                dq = np.array(DQ)[None, :]
                M1 = m * fa + 0.0 * dq
                M2 = m * (1.0 - fa) + 0.0 * dq
                # ★★整数に丸めてから足す。★これで親も子も必ず整数のまま
                QA = np.round(Qp * fa) + dq
                QB = Qp - QA
                floor = MFRAG_MIN if Q_ON else 1.0
                E = (binding(M1, QA) + binding(M2, QB)
                     - float(binding(np.array([m]), np.array([Qp]))[0]))
                E = np.where(np.minimum(M1, M2) < floor, -np.inf, E)
                k = int(np.argmax(E))
                if not np.isfinite(E.flat[k]) or E.flat[k] <= 0.0:
                    continue
                bdE = float(E.flat[k])
                m1, m2 = float(M1.flat[k]), float(M2.flat[k])
                qa_, qb_ = float(QA.flat[k]), float(QB.flat[k])
                dE = bdE
                u = self.jit.take(DIM) - 0.5
                u /= (np.linalg.norm(u) + 1e-9)
                # ★★破片は親の速度をそのまま（★運動量保存）。★出た分は全部まわりへ
                newp.append(self.pos[idx] + u * 0.6)
                newv.append(self.vel[idx].copy())
                newm.append(m1)
                newq.append(qa_)
                newid.append(self.new_pid())
                newp.append(self.pos[idx] - u * 0.6)
                newv.append(self.vel[idx].copy())
                newm.append(m2)
                newq.append(qb_)
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
            self.q = np.concatenate([self.q[keep], np.array(newq)])
            assert len(self.pid) == len(self.mass), '名札の数が合わない'
            assert len(self.q) == len(self.mass), '電荷の数が合わない'
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
        Ebind = float(binding(self.mass, self.q).sum())   # ★質量に残っている燃料
        Ekin = 0.5 * float((self.mass * (self.vel * self.vel).sum(1)).sum())
        return Ebind, Ekin

    def Wq_step(self):
        w = self.Wq - self._wqprev
        self._wqprev = self.Wq
        return w

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
        gi, grads, qgr = self.field()
        for ax in range(DIM):
            f[:, ax] += GRAV * grads[ax][gi]
        # ★★★電気 ── ★同符号は反発（★だから重力と符号が逆）。★加速度は質量で割る
        if Q_ON:
            qk = KQ * self.q / np.maximum(self.mass, 1e-9)
            for ax in range(DIM):
                f[:, ax] -= qk * qgr[ax][gi]

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
        qacc = np.zeros_like(f1)
        if Q_ON:
            qk = KQ * self.q / np.maximum(self.mass, 1e-9)
            for ax in range(DIM):
                qacc[:, ax] = -qk * qgr[ax][gi1]
        # ★★★重力がした仕事を数える（2026-09-11）。
        #   ★前はこれを台帳に入れておらず、★★**湧いているように見えていた**（実測 +2991）。
        #   ★実際は「潰れると熱くなる」という**本物の物理**だった。
        gold = np.zeros_like(f)
        for ax in range(DIM):
            gold[:, ax] = GRAV * grads[ax][gi]
        self.Wgrav += float((self.mass[:, None] * 0.5 * (gold + gacc) * dx).sum())
        # ★★★電気がした仕事。★重力のときに入れ忘れて +2991 湧いて見えた。同じ轍は踏まない
        if Q_ON:
            qold = np.zeros_like(f)
            qk = KQ * self.q / np.maximum(self.mass, 1e-9)
            for ax in range(DIM):
                qold[:, ax] = -qk * qgr[ax][gi]
            self.Wq += float((self.mass[:, None] * 0.5 * (qold + qacc) * dx).sum())
            f1 = f1 + qacc
        f1 = f1 + gacc
        self.vel = self.vel + 0.5 * f1 * DT         # ★残り半分を蹴る
        k1 = self._ke()
        self.d_move += ((k1 - k0) + (self.Elj - lj0)
                        - self.Wgrav_step() - self.Wq_step())

        # ── ★② 変換 ─────────────────────────
        k0 = self._ke()
        b0 = float(binding(self.mass, self.q).sum())
        lj_before = self.Elj                      # ★★変換の前の LJ 位置エネルギー
        self.transform(r2, s2)
        self.forces()                             # ★★σ が変わったので測り直す
        lj_after = self.Elj
        k1 = self._ke()
        b1 = float(binding(self.mass, self.q).sum())
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
            self.Eout -= float(binding(self.mass[~keep], self.q[~keep]).sum())
            self.q_esc += float(self.q[~keep].sum())
            self.pos, self.vel, self.mass = (self.pos[keep], self.vel[keep],
                                             self.mass[keep])
            self.pid = self.pid[keep]
            self.q = self.q[keep]
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
        if Q_ON:
            _atom = self.mass >= 1.0
            cut = cut & _atom[:, None] & _atom[None, :]
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
        Ebind = float(binding(m, self.q).sum())      # ★質量の中に残っている燃料
        Ekin = 0.5 * float((m * v2).sum())           # ★運動
        # ★★★台帳の全部（2026-09-11 に2度直した）:
        #   ★結合エネルギーは**負の位置エネルギー**（符号）
        #   ★★**LJ の位置エネルギー**と**重力がした仕事**が抜けていた
        Etot = Ekin + self.Elj + self.Eout - Ebind - self.Wgrav - self.Wq
        if self.E0tot is None:
            self.E0tot = Etot
        # ★★★ずれの割り方（2026-09-11 の訂正）。
        #   ★前は**初期値**で割っていた。★でも初期値は −312 しかなく（★全部が質量1）、
        #   ★★**ゼロに近い数で割っていた**ので、★絶対値 +318 が「+102%」に見えていた。
        #   ★★★正しくは**流れているエネルギーの大きさ**で割る。
        scale = (abs(Ekin) + abs(self.Elj) + abs(self.Eout) + abs(Ebind)
                 + abs(self.Wgrav) + abs(self.Wq))
        drift = (Etot - self.E0tot) / max(1.0, scale)
        drift_abs = Etot - self.E0tot

        # ── ★質量の台帳 ────────────────────────
        mnow = float(m.sum())
        mdrift = (mnow + self.mass_esc - self.mass0) / max(1e-9, self.mass0)

        # ── ★★★電荷の台帳（2026-09-11）── 3本目の保存則 ────────
        #   ★はじめは全部ゼロ。★分裂で ＋ と − に分かれても、★合計は動いてはいけない。
        #   ★合計がゼロから離れたら、★そこに嘘がある。
        qnow = float(self.q.sum())
        qdrift = qnow + self.q_esc - self.q0
        qpos = float(self.q[self.q > 0].sum())
        # ★★★「電子」＝ 軽くて負のもの。★僕は電子を定義していない。
        #   ★★この2つの条件はどちらも曲線が作った結果。★名前を付けているだけ。
        light = m < 0.5
        nele = int((light & (self.q < 0)).sum())
        # ★★★電子は核に捕まっているか（2026-09-11）。
        #   ★★でたらめに置いた場合と比べる。★1より小さければ**寄っている**。
        #   ★これが無いと「クーロン力を入れた」だけで何も言えない。
        bound_r = 1.0
        pos_at = self.pos[(m >= 1.0) & (self.q > 0)]
        if nele and len(pos_at) >= 2:
            pe = self.pos[light & (self.q < 0)]
            dd = np.sqrt(((pe[:, None, :] - pos_at[None, :, :]) ** 2).sum(-1))
            near = float(dd.min(1).mean())
            # ★★対照: 同じ雲の中にでたらめに置いた点から、最も近い正の原子まで
            rr_ = np.sqrt(((pos_at - self.c) ** 2).sum(1))
            R_ = max(1e-6, float(rr_.max()))
            u = self.jit.take(nele * DIM).reshape(nele, DIM) - 0.5
            u = u / np.maximum(1e-9, np.sqrt((u * u).sum(1))[:, None])
            rad_ = R_ * (self.jit.take(nele) ** (1.0 / DIM))[:, None]
            pr = self.c + u * rad_
            d2 = np.sqrt(((pr[:, None, :] - pos_at[None, :, :]) ** 2).sum(-1))
            ctl = float(d2.min(1).mean())
            bound_r = near / max(1e-9, ctl)

        # ★★★同位体（2026-09-11）。★電荷が元素の身元、★質量が同位体。
        #   ★電荷が在る前は「質量＝種類」でよかったが、★いまは本物と同じく電荷が身元。
        #   ★★正直に: この世界の電荷は**飛び飛びではない**。★整数に丸めて数えている。
        #   ★どれだけ整数に寄っているかも一緒に出す（★でたらめなら 20%）。
        elems = []
        nint = 0.0
        if Q_ON:
            sel = (m >= 1.0) & (self.q > 0)
            if sel.any():
                qa2, ma2 = self.q[sel], m[sel]
                nint = float(np.mean(np.abs(qa2 - np.round(qa2)) < 0.1))
                for k in sorted(set(np.round(qa2).astype(int).tolist())):
                    pick = np.round(qa2).astype(int) == k
                    mm2 = np.round(ma2[pick], 1)
                    elems.append(dict(
                        q=int(k), n=int(pick.sum()),
                        iso=int(len(set(mm2.tolist()))),
                        med=round(float(np.median(mm2)), 1),
                        want=round(k / max(1e-9, Z_FRAC), 1),   # ★谷の予想
                        lo=round(float(mm2.min()), 1),
                        hi=round(float(mm2.max()), 1)))
                elems = elems[:20]
        m_ele = float(m[light & (self.q < 0)].mean()) if nele else 0.0
        q_ele = float(self.q[light & (self.q < 0)].mean()) if nele else 0.0

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
            qdrift=round(qdrift, 6), qpos=round(qpos, 2),
            nele=nele, mele=round(m_ele, 5), qele=round(q_ele, 3),
            bound=round(bound_r, 3), elems=elems, qint=round(nint, 3),
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
            Wq=round(self.Wq, 1),
            Elj=round(self.Elj, 1))

    # ── ★★★続きのための紙 ────────────────────────
    #   ★保存するのは**世界そのもの**（位置・速さ・質量・にじみの場）と
    #   ★**台帳**（湧いた分・逃げた分・外へ出た分）と**指紋**（再訪の記憶）。
    #   ★ジッタは保存しない ── ★崩壊の「いつ」は毎回あたらしく実行時間から取る。
    SCALARS = ("step_n", "secs", "revisits", "escaped", "nfuse", "nfiss", "nfuse_endo",
               "q0", "q_esc", "Wq", "_wqprev",
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
            pid=self.pid, next_pid=np.array([self.next_pid]), q=self.q,
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
        self.q = z["q"] if "q" in z.files else np.zeros(len(self.mass))
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
                  " | ★電荷ずれ%+.4f ＋%6.1f 電子%4d(質量%.4f 電荷%+.2f 核への寄り%.2f)"
                  % (st["step"], st["T"], st["Tin"], st["Tout"], st["grad"],
                     st["fuel"], st["Ekin"], st["Eout"], st["drift"], st["dabs"],
                     st["mass"], st["mesc"], st["mdrift"], st["kinds"],
                     st["shannon"], st["mmax"], st["mmed"], st["layers"],
                     st["mscales"], st["fuse"], st["endo"], st["fiss"],
                     st["n"], st["esc"], st["revisit"], st["newness"],
                     st["mol"], st["molmax"], st["molkinds"], st["bonds"],
                     st["mollife"], st["molbest"],
                     st["qdrift"], st["qpos"], st["nele"], st["mele"], st["qele"],
                     st["bound"]),
                  flush=True)
            print("      ★★湧いた内訳: 動き%+9.1f 変換%+7.1f 配る%+9.1f | "
                  "重力の仕事%+10.1f  電気の仕事%+10.1f  LJ位置%+9.1f"
                  % (st["dmove"], st["dtr"], st["ddel"], st["Wg"], st["Wq"], st["Elj"]),
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
