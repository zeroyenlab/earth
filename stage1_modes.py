# -*- coding: utf-8 -*-
"""★段1 ── 箱に閉じ込めた波から、★「種類」が出てくるか。（★2回目・測り方を直した）

★★1回目に壊れていた所（★全部こちらの設計ミス）
   ★① 離散性を「上位6つの山の高さの割合」で測った → ★山の数を6に固定していたので、
        ★**どんな波形でも同じような値**になった。★★陰性対照の方が高く出た。
        → ★**スペクトル平坦度**で測る。★白色雑音なら1に近く、★線スペクトルなら0に近い。
   ★② 叩く場所をばらばらにした → ★**鳴るモードが毎回変わる**（物理として当たり前）。
        → ★**1点を1回だけ叩く**（デルタ）。★これで全モードが等しく鳴る。
        → ★指紋は「上位」でなく、★**3回の共通部分**を取る。
   ★③ 箱の外に空間を置いていなかった → ★イベント駆動の節約は**外でしか出ない**。
        → ★★広い場の真ん中に小さい箱を置く。
   ★④ 答え合わせが無かった → ★**四角なら理論値が分かる**（√(n²+m²)）。★突き合わせる。

★★★僕が決めるのは形だけ。★振動数は1つも指定しない。
"""
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N = 128
STEPS = 16384
C2 = 0.20
DAMP = 0.99985
EPS = 2e-4


def shape(kind):
    """★箱（True の所を波が通れる）。★広い場の真ん中に置く。"""
    m = np.zeros((N, N), bool)
    c = N // 2
    if kind == "四角24":
        m[c-12:c+12, c-12:c+12] = True
    elif kind == "四角32":
        m[c-16:c+16, c-16:c+16] = True
    elif kind == "長方形":
        m[c-8:c+8, c-20:c+20] = True
    elif kind == "L字":
        m[c-16:c+4, c-16:c-2] = True
        m[c-8:c+4, c-16:c+16] = True
    elif kind == "丸":
        y, x = np.ogrid[:N, :N]
        m = (y-c)**2 + (x-c)**2 < 14**2
    elif kind == "三角":
        for i in range(26):
            m[c-13+i, c-i:c+i+1] = True
    elif kind == "箱なし":
        m[:, :] = True
    return m


def ring(mask, open_field, hit, probe, event=False):
    """★1点を1回だけ叩いて、1点の揺れを記録する。"""
    u = np.zeros((N, N))
    up = np.zeros((N, N))
    u[hit] = 1.0                     # ★★デルタ。★全モードを等しく鳴らす

    absorb = np.ones((N, N))
    if open_field:                   # ★★陰性対照 ── 端から逃がす
        for i in range(18):
            f = 0.88 + 0.12 * i / 18.0
            absorb[i, :] = np.minimum(absorb[i, :], f)
            absorb[-1-i, :] = np.minimum(absorb[-1-i, :], f)
            absorb[:, i] = np.minimum(absorb[:, i], f)
            absorb[:, -1-i] = np.minimum(absorb[:, -1-i], f)

    rec = np.zeros(STEPS)
    touched = 0
    for t in range(STEPS):
        lap = (np.roll(u, 1, 0) + np.roll(u, -1, 0)
               + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4.0 * u)
        un = (2.0 * u - up + C2 * lap) * DAMP
        un *= mask
        un *= absorb
        if event:
            live = (np.abs(un) > EPS) | (np.abs(un - u) > EPS)
            live = (live | np.roll(live, 1, 0) | np.roll(live, -1, 0)
                    | np.roll(live, 1, 1) | np.roll(live, -1, 1))
            un = np.where(live, un, 0.0)
            touched += int(live.sum())
        else:
            touched += N * N        # ★★全部やったら（★場の全体）
        up, u = u, un
        rec[t] = u[probe]
    return rec, touched, STEPS * N * N


def spec(rec):
    w = np.hanning(len(rec))
    sp = np.abs(np.fft.rfft(rec * w)) ** 2
    sp[:6] = 0.0
    s = sp.sum()
    return sp / s if s > 0 else sp


def flatness(sp):
    """★スペクトル平坦度。★1に近い=白色（連続）、0に近い=線（離散）"""
    p = sp[sp > 1e-15]
    if len(p) < 10:
        return 1.0
    g = np.exp(np.log(p).mean())     # 幾何平均
    a = p.mean()                     # 算術平均
    return float(g / a)


def lines(sp, rel=0.05):
    """★山の位置。★一番高い山の rel 倍を超えた極大だけ"""
    th = sp.max() * rel
    return [i for i in range(2, len(sp)-2)
            if sp[i] > th and sp[i] > sp[i-1] and sp[i] >= sp[i+1]]


def main():
    kinds = ["四角24", "四角32", "長方形", "L字", "丸", "三角", "箱なし"]
    c = N // 2
    print("★★★段1（2回目）── 箱の中の波から「種類」が出るか")
    print("   ★決めたのは**形だけ**。振動数も、どのモードが鳴るかも決めていない。")
    print("   ★★陰性対照「箱なし」で離散が出たら、★測り方が壊れている。\n")

    print("── ① 離散が出るか（★平坦度: 1=連続 / 0=離散）──")
    print("%-8s %-9s %-8s %s" % ("形", "平坦度", "線の数", "低い方の線"))
    keep = {}
    for kd in kinds:
        m = shape(kd)
        ins = np.argwhere(m)
        hit = tuple(ins[len(ins)//3])
        probe = tuple(ins[len(ins)//2 + 7])
        rec, _, _ = ring(m, kd == "箱なし", hit, probe)
        sp = spec(rec)
        f, L = flatness(sp), lines(sp)
        keep[kd] = (f, L, sp)
        print("%-8s %-9.4f %-8d %s" % (kd, f, len(L), L[:7]))

    ob = keep["箱なし"][0]
    bx = [keep[k][0] for k in kinds if k != "箱なし"]
    print("\n   ★箱なし %.4f  ／  ★★箱あり 平均 %.4f  → %s"
          % (ob, float(np.mean(bx)),
             "★★★箱の方が離散" if np.mean(bx) < ob else "★★駄目。箱が効いていない"))

    print("\n── ② 理論と合うか（★四角24。★理論値 √(n²+m²) の比）──")
    sp = keep["四角24"][2]
    L = keep["四角24"][1]
    if L:
        f0 = L[0]
        got = [round(x / f0, 3) for x in L[:8]]
        th = sorted({np.sqrt(n*n + mm*mm) for n in range(1, 6) for mm in range(1, 6)})
        th = [round(x / th[0], 3) for x in th[:8]]
        print("   実測の比 %s" % got)
        print("   理論の比 %s" % th)

    print("\n── ③ 叩き方を変えても、同じ種類か（★3回の共通部分）──")
    print("%-8s %-9s %s" % ("形", "共通の線", "一致率"))
    for kd in kinds:
        if kd == "箱なし":
            continue
        m = shape(kd)
        ins = np.argwhere(m)
        sets = []
        for q in (3, 5, 8):
            hit = tuple(ins[len(ins)//q])
            probe = tuple(ins[len(ins)//2 + q])
            rec, _, _ = ring(m, False, hit, probe)
            sets.append(set(lines(spec(rec))))
        inter = set.intersection(*sets)
        union = set.union(*sets)
        print("%-8s %-9d %.0f%%   %s"
              % (kd, len(inter), 100.0*len(inter)/max(1, len(union)),
                 sorted(inter)[:6]))

    print("\n── ④ 形が違えば種類が違うか（★共通の線どうしを比べる）──")
    fp = {}
    for kd in kinds:
        if kd == "箱なし":
            continue
        m = shape(kd)
        ins = np.argwhere(m)
        sets = []
        for q in (3, 5, 8):
            hit = tuple(ins[len(ins)//q])
            probe = tuple(ins[len(ins)//2 + q])
            sets.append(set(lines(spec(ring(m, False, hit, probe)[0]))))
        fp[kd] = frozenset(set.intersection(*sets))
    ks = list(fp)
    dup = 0
    for i in range(len(ks)):
        for j in range(i+1, len(ks)):
            a, b = fp[ks[i]], fp[ks[j]]
            ov = len(a & b) / max(1, len(a | b))
            if ov > 0.5:
                dup += 1
                print("   ★★かぶり: %s と %s（重なり %.0f%%）" % (ks[i], ks[j], ov*100))
    print("   ★形 %d 種類 → ★**違う種類 %d**（かぶり %d 組）"
          % (len(ks), len(set(fp.values())), dup))

    print("\n── ⑤ イベント駆動で安くなるか（★広い場に小さい箱）──")
    print("%-8s %-13s %-13s %-8s %s" % ("形", "更新したセル", "全部やったら", "節約", "指紋"))
    for kd in ("四角24", "L字", "丸"):
        m = shape(kd)
        ins = np.argwhere(m)
        hit = tuple(ins[len(ins)//3]); probe = tuple(ins[len(ins)//2 + 7])
        r1, t1, full = ring(m, False, hit, probe, event=True)
        r2, _, _ = ring(m, False, hit, probe, event=False)
        s1, s2 = set(lines(spec(r1))), set(lines(spec(r2)))
        ov = len(s1 & s2) / max(1, len(s1 | s2))
        print("%-8s %-13d %-13d %-8.1f%% %s"
              % (kd, t1, full, 100.0*(1 - t1/full),
                 "★同じ(%.0f%%)" % (ov*100) if ov > 0.7 else "★★壊れた(%.0f%%)" % (ov*100)))


if __name__ == "__main__":
    main()
