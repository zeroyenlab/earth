# -*- coding: utf-8 -*-
"""★★★量子 ── 電子の居場所を、にじみで解く。

★★★新しい法則は足していない。
   ★シュレーディンガーの式は、★**にじみの式の係数を虚数にしたもの**。
   　にじみ　　　　∂ρ/∂t = D ∇²ρ
   　量子　　　　 iħ ∂ψ/∂t = −(ħ²/2m)∇²ψ + Vψ
   　★τ = it/ħ と置くと → ∂ψ/∂τ = (ħ²/2m)∇²ψ − Vψ
   ★★これは**吸い込み項つきの、にじみの式**。★世界には既にその機械がある。

★★★虚時間で流すと、★エネルギーの高い成分ほど速く減る。
   → ★放っておくだけで**一番低い状態に落ちる**。★解く作業が要らない。

★★★足すのは法則ではなく、**2つの縛り**だけ:
   ① 各 ψ の合計を 1 に保つ　　→ ★電子1個が1個であること
   ② ψ どうしを直交させる　　　→ ★**パウリ原理**（★同じ状態に入れない）
   ★★★スピンは本物に合わせて「1つの形に2個まで」。

★単位は原子単位（ħ = 電子の質量 = 電気の強さ = 1）。★水素の基底は −0.5 のはず。

★自分で確かめる: python quantum.py
"""
import numpy as np


class Box(object):
    """★かたまり1つを覆う小さい格子。★世界ぜんぶを覆うのは無理（★1億点になる）。"""

    def __init__(self, centers, pad=6.0, g=40):
        centers = np.atleast_2d(np.asarray(centers, dtype=float))
        lo = centers.min(0) - pad
        hi = centers.max(0) + pad
        self.g = int(g)
        self.lo, self.hi = lo, hi
        ax = [np.linspace(lo[k], hi[k], self.g) for k in range(3)]
        self.dx = float(np.mean([a[1] - a[0] for a in ax]))
        self.X, self.Y, self.Z = np.meshgrid(*ax, indexing="ij")
        self.dv = self.dx ** 3

    def potential(self, centers, charges, soft=None):
        """★核が作る井戸。★中心で無限に落ちないよう、★格子の刻みぶんだけ丸める。

        ★★丸めないと、★格子の1点だけが −∞ になって**答えが刻みに振り回される**。
        """
        if soft is None:
            soft = 0.55 * self.dx
        V = np.zeros_like(self.X)
        for c, z in zip(np.atleast_2d(centers), np.atleast_1d(charges)):
            r2 = ((self.X - c[0]) ** 2 + (self.Y - c[1]) ** 2
                  + (self.Z - c[2]) ** 2)
            V -= z / np.sqrt(r2 + soft * soft)
        return V


def wall(p):
    """★★★箱の壁。★★ψ そのものを端でゼロにする。

    ★★★2026-09-11 の訂正: 前は**∇²ψ の方**を端でゼロにしていた。
    　★それは「壁」ではなく「自由な端」で、★しかも演算子が対称でなくなる。
    　★実測: 格子を細かくするほど答えが悪化した（−0.41 → −0.32 → −0.087）。
    ★ψ が端でゼロなら、★roll が反対側から持ってくるのもゼロ。★矛盾しない。
    """
    for ax in range(3):
        sl = [slice(None)] * 3
        sl[ax] = 0
        p[tuple(sl)] = 0.0
        sl[ax] = -1
        p[tuple(sl)] = 0.0
    return p


def laplacian(p, dx):
    """★★にじみと同じ形（★7点）。"""
    out = -6.0 * p
    for ax in range(3):
        out += np.roll(p, 1, ax) + np.roll(p, -1, ax)
    return out / (dx * dx)


def _ortho(psi, dv):
    """★★★パウリ原理。★互いに直交させ、★合計を1にする（グラム・シュミット）。

    ★これが「同じ状態には入れない」ということ。★表は書いていない。
    """
    for i in range(len(psi)):
        for j in range(i):
            psi[i] -= float((psi[i] * psi[j]).sum() * dv) * psi[j]
        n = float((psi[i] * psi[i]).sum() * dv)
        psi[i] /= np.sqrt(max(n, 1e-300))
    return psi


def rayleigh_ritz(psi, V, dx, dv):
    """★★★いま持っている形の張る空間の中で、★一番低い組に取り替える。

    ★虚時間で流すだけでは遅い（★差が小さい段ほど分かれない）。
    ★★これは新しい物理ではない ── ★**同じ空間の中で見方を変えるだけ**。
    """
    n = len(psi)
    hp = np.array([-0.5 * laplacian(p, dx) + V * p for p in psi])
    H = np.array([[float((psi[i] * hp[j]).sum() * dv) for j in range(n)]
                  for i in range(n)])
    H = 0.5 * (H + H.T)
    w, U = np.linalg.eigh(H)
    return np.einsum("ji,j...->i...", U, psi), w


def energies(psi, V, dx, dv):
    """★★各 ψ のエネルギー ⟨ψ| −½∇² + V |ψ⟩"""
    out = []
    for p in psi:
        hp = -0.5 * laplacian(p, dx) + V * p
        out.append(float((p * hp).sum() * dv))
    return np.array(out)


def solve(box, centers, charges, norb, psi=None, sweeps=400, tol=1e-7,
          jit=None):
    """★★★にじみを虚時間で流して、★低い方から norb 個の居場所を出す。

    ★psi を渡すと**そこから続ける**（★核が少し動いただけなら数回で済む）。
    """
    V = box.potential(centers, charges)
    g, dx, dv = box.g, box.dx, box.dv
    if psi is None or len(psi) != norb or psi[0].shape[0] != g:
        # ★★★でたらめから始める。★種は本物のジッタから取る（★世界の流儀）
        if jit is not None:
            raw = jit.take(norb * g * g * g).reshape(norb, g, g, g) - 0.5
        else:
            raw = np.random.RandomState(0).rand(norb, g, g, g) - 0.5
        psi = np.asarray(raw, dtype=float)
        # ★★核の近くに寄せておくだけ（★形は与えない。★深さの平方根で重みを付ける）
        psi = psi * (np.abs(V) ** 0.5)[None, :, :, :]
    psi = np.array(psi, dtype=float, copy=True)
    for i in range(len(psi)):
        psi[i] = wall(psi[i])
    psi = _ortho(psi, dv)

    # ★★刻み幅。★運動の項と井戸の深さ、★両方で安定でなければならない
    dt_kin = 0.4 * dx * dx / 3.0
    dt_pot = 0.4 / max(1e-9, float(np.abs(V).max()))
    dt = min(dt_kin, dt_pot)

    prev = None
    for it in range(sweeps):
        for i in range(norb):
            hp = -0.5 * laplacian(psi[i], dx) + V * psi[i]
            psi[i] = wall(psi[i] - dt * hp)
        psi = _ortho(psi, dv)
        if it % 10 == 9:
            psi, e = rayleigh_ritz(psi, V, dx, dv)
            psi = _ortho(psi, dv)
            if prev is not None and np.max(np.abs(e - prev)) < tol:
                break
            prev = e
    psi, e = rayleigh_ritz(psi, V, dx, dv)
    return _ortho(psi, dv), e, V


def occupy(norb_e):
    """★★★スピン ── 本物に合わせて「1つの形に2個まで」。

    ★★電子の数から、★何個の形が要るか／各形に何個入るかを返す。
    """
    n = int(norb_e)
    need = (n + 1) // 2
    occ = [2] * (n // 2) + ([1] if n % 2 else [])
    return need, np.array(occ, dtype=float)


# ── ★★★自分で確かめる ────────────────────────────
def selftest():
    print("★★★量子の自己テスト（★本物の答えと突き合わせる）")
    print()
    print("① 水素（陽子1個・電子1個）── ★本物の答えは **-0.5**")
    print("%8s %8s %10s %10s %10s" % ("格子", "刻み", "基底", "本物", "ずれ"))
    ok = True
    for g, pad in ((32, 8.0), (48, 9.0), (64, 10.0)):
        b = Box([[0.0, 0.0, 0.0]], pad=pad, g=g)
        c = np.array([[0.0, 0.0, 0.0]])
        _p, e, _V = solve(b, c, [1.0], 1, sweeps=600)
        err = abs(e[0] - (-0.5))
        print("%8d %8.3f %10.4f %10.4f %10.4f" % (g, b.dx, e[0], -0.5, err))
    print()
    print("② 水素の上の段 ── ★本物は 2s/2p が **-0.125**（★4本とも同じ）")
    b = Box([[0.0, 0.0, 0.0]], pad=14.0, g=64)
    c = np.array([[0.0, 0.0, 0.0]])
    _p, e, _V = solve(b, c, [1.0], 5, sweeps=1200)
    print("   出た段:", " ".join("%.4f" % x for x in e))
    print("   本物  : -0.5000 " + " ".join(["-0.1250"] * 4))
    print()
    print("③ 電荷2の核 ── ★本物は 1s が **-2.0**（★Z²倍）")
    b = Box([[0.0, 0.0, 0.0]], pad=6.0, g=48)
    _p, e, _V = solve(b, c, [2.0], 1, sweeps=800)
    print("   出た: %.4f  / 本物: -2.0000  / ずれ %.4f" % (e[0], abs(e[0] + 2.0)))
    print()
    print("④ パウリ（直交しているか）── ★内積はゼロのはず")
    b = Box([[0.0, 0.0, 0.0]], pad=10.0, g=48)
    p, e, _V = solve(b, c, [1.0], 4, sweeps=600)
    mx = 0.0
    for i in range(len(p)):
        for j in range(i):
            mx = max(mx, abs(float((p[i] * p[j]).sum() * b.dv)))
    print("   一番大きい内積: %.2e （★ゼロならパウリが効いている）" % mx)
    print()
    print("⑤ スピン ── ★1つの形に2個まで")
    for n in (1, 2, 3, 4, 10, 11):
        need, occ = occupy(n)
        print("   電子%2d個 → 形%2d本 / 入り方 %s" % (n, need, list(occ.astype(int))))
    return ok


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    selftest()
