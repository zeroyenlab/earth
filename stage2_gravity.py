# -*- coding: utf-8 -*-
"""★段2 ── にじみから、力が出るか。

★★測る問い
   ① ★**べき則になるか**（★指数関数ではなく）
   ② ★★**指数が、次元の言う通りか**
        ★2次元 → 力 ∝ 1/r¹（★ポテンシャルは log r）
        ★3次元 → 力 ∝ 1/r²（★ポテンシャルは 1/r）
        ★★★**僕は 1/r² と書かない。★にじませるだけ。★次元が決める。**
   ③ ★**2体は本当に引き合うか**

★★★陰性対照: ★**減衰つきのにじみ**（湯川型）。
   ★これは指数関数で遮蔽されるので、★**べき則にならないはず**。
   ★ここでべきが出たら、★**測り方が壊れている**（何でもべきに見えている）。
"""
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def lap(a):
    s = -2 * a.ndim * a
    for ax in range(a.ndim):
        s = s + np.roll(a, 1, ax) + np.roll(a, -1, ax)
    return s


def relax(shape_, src, decay=0.0, iters=12000, tol=1e-7):
    """★にじみを落ち着かせる。★端は吸う（ゼロ固定）。

    ★★2026-09-11 に直した2点:
       ★① **安定限界**。★d次元の陽的拡散は D <= 1/(2d)。
          ★3次元で 0.18 を使っていた（限界 0.167）→ **発散していた**。
       ★② **収束の確認**。★回数で打ち切ると、★にじみが届いていない所まで測ってしまう。
          → ★変化が止まるまで回す。
    """
    D = 1.0 / (2.0 * len(shape_)) * 0.9        # ★安定限界の9割
    m = np.zeros(shape_)
    for it in range(iters):
        nm = m + D * lap(m) + src - decay * m
        for ax in range(nm.ndim):
            sl = [slice(None)] * nm.ndim
            sl[ax] = 0;  nm[tuple(sl)] = 0.0
            sl[ax] = -1; nm[tuple(sl)] = 0.0
        d = np.abs(nm - m).max()
        m = nm
        if d < tol * max(1e-12, np.abs(m).max()):
            return m, it + 1
    return m, iters


def radial(m, c):
    """★中心からの距離ごとに、ポテンシャルと力の大きさを出す。"""
    idx = np.indices(m.shape).astype(float)
    r = np.sqrt(sum((idx[k] - c[k]) ** 2 for k in range(m.ndim)))
    g = np.gradient(m)
    f = np.sqrt(sum(gi ** 2 for gi in g))
    # ★★測る範囲は、端の影響が入らない所だけ
    lo, hi = 4.0, min(m.shape) / 3.0
    rs, ps, fs = [], [], []
    for rr in np.arange(lo, hi, 1.0):
        sel = (r >= rr - 0.5) & (r < rr + 0.5)
        if sel.sum() < 6:
            continue
        rs.append(rr)
        ps.append(float(np.abs(m[sel]).mean()))
        fs.append(float(f[sel].mean()))
    return np.array(rs), np.array(ps), np.array(fs)


def slope(r, y):
    """★log-log の傾き ＝ べきの指数"""
    ok = (y > 0) & (r > 0)
    if ok.sum() < 5:
        return float("nan"), 0.0
    lx, ly = np.log(r[ok]), np.log(y[ok])
    a, b = np.polyfit(lx, ly, 1)
    pred = a * lx + b
    ss = 1.0 - ((ly - pred) ** 2).sum() / max(1e-12, ((ly - ly.mean()) ** 2).sum())
    return float(a), float(ss)


def run(dim, decay, n):
    shape_ = (n,) * dim
    c = tuple(n // 2 for _ in range(dim))
    src = np.zeros(shape_)
    src[c] = 1.0
    m, its = relax(shape_, src, decay)
    r, p, f = radial(m, c)
    # ★2次元のポテンシャルは log r なので、★べきでなく log で当てる
    logfit = None
    if dim == 2:
        ok = p > 0
        a, b = np.polyfit(np.log(r[ok]), p[ok], 1)
        pred = a * np.log(r[ok]) + b
        r2 = 1.0 - ((p[ok]-pred)**2).sum() / max(1e-12, ((p[ok]-p[ok].mean())**2).sum())
        logfit = (float(a), float(r2))
    return slope(r, p), slope(r, f), r, p, f, its, logfit


def main():
    print("★★★段2 ── にじみから力が出るか")
    print("   ★僕が書いたのは「隣へにじむ」だけ。★1/r² とは書いていない。\n")

    print("── ① ② べき則になるか／指数は次元の言う通りか ──")
    print("%-22s %-16s %-16s" % ("条件", "ポテンシャルの指数", "力の指数"))
    rows = []
    # ★★格子は小さくてよい。★収束まで回す方が大事（★大きいと収束しない）
    for dim, n, tag, decay in ((2, 64, "2次元・にじみ", 0.0),
                               (3, 34, "3次元・にじみ", 0.0),
                               (2, 64, "★陰性: 2次元・減衰つき", 0.02),
                               (3, 34, "★陰性: 3次元・減衰つき", 0.02)):
        (pa, pr2), (fa, fr2), r, p, f, its, lf = run(dim, decay, n)
        rows.append((tag, dim, decay, pa, pr2, fa, fr2, lf))
        extra = ""
        if lf is not None:
            extra = "  ／ log当て R²%.3f" % lf[1]
        print("%-22s %7.3f (R²%.3f) %7.3f (R²%.3f)  %5d回で収束%s"
              % (tag, pa, pr2, fa, fr2, its, extra))

    print("\n   ★理論: 2次元 → ポテンシャル log r（指数なし）・力 **−1**")
    print("         3次元 → ポテンシャル **−1**・力 **−2**")

    print("\n── ★判定 ──")
    for tag, dim, decay, pa, pr2, fa, fr2, lf in rows:
        want = -1.0 if dim == 2 else -2.0
        if decay > 0:
            ok = fr2 < 0.95
            print("  %-22s %s（R² %.3f。★べきなら高いはず）"
                  % (tag, "★★陰性対照が正しく外れた" if ok else "★★駄目。測り方が壊れている", fr2))
        else:
            ok = abs(fa - want) < 0.25 and fr2 > 0.95
            print("  %-22s 力の指数 %.3f ／ 理論 %.1f → %s"
                  % (tag, fa, want, "★★★合った" if ok else "★★ずれた"))

    print("\n── ③ 2体は引き合うか（3次元）──")
    n, dim = 40, 3
    c = n // 2
    pos = [np.array([c - 8.0, float(c), float(c)]),
           np.array([c + 8.0, float(c), float(c)])]
    d0 = np.linalg.norm(pos[0] - pos[1])
    for step in range(24):
        src = np.zeros((n,) * dim)
        for q in pos:
            src[tuple(np.round(q).astype(int))] = 1.0
        m, _ = relax((n,) * dim, src, 0.0, iters=250, tol=1e-5)
        g = np.gradient(m)
        for k, q in enumerate(pos):
            ix = tuple(np.round(q).astype(int))
            # ★★にじみの濃い方へ動く（★勾配を上る＝引力）
            v = np.array([gi[ix] for gi in g])
            nv = np.linalg.norm(v)
            if nv > 0:
                pos[k] = q + 1.1 * v / nv
        if step % 10 == 9:
            print("   %2d歩   距離 %.2f" % (step + 1, np.linalg.norm(pos[0] - pos[1])))
    d1 = np.linalg.norm(pos[0] - pos[1])
    print("   ★はじめ %.2f → いま %.2f  → %s"
          % (d0, d1, "★★★引き合った" if d1 < d0 - 1.0 else "★★引き合わなかった"))


if __name__ == "__main__":
    main()
