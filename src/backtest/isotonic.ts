/**
 * 単調（非減少）キャリブレーション。
 * 生の win_prob → 実勝率 の対応を、等頻度ビン＋PAV(pool adjacent violators)で単調関数に整形。
 * softmax温度では全確率帯を同時に較正できないため（低確率帯は過信のまま）、
 * ここで生確率→較正確率の写像を学習し、EV判定に使う。
 */

export interface Calibration {
  /** ビン代表の生確率（昇順）。 */
  x: number[];
  /** 対応する較正確率（非減少）。 */
  y: number[];
}

interface Pt {
  p: number; // 生確率
  win: number; // 0/1
}

/** 学習: (生確率, 勝敗) から単調較正を作る。bins=等頻度ビン数。 */
export function fitIsotonic(points: Pt[], bins = 25): Calibration {
  const pts = [...points].sort((a, b) => a.p - b.p);
  const n = pts.length;
  if (n === 0) return { x: [0, 1], y: [0, 1] };
  const per = Math.max(1, Math.floor(n / bins));
  const xs: number[] = [];
  const ys: number[] = [];
  const ws: number[] = [];
  for (let i = 0; i < n; i += per) {
    const chunk = pts.slice(i, i + per);
    if (chunk.length === 0) continue;
    xs.push(chunk.reduce((a, c) => a + c.p, 0) / chunk.length);
    ys.push(chunk.reduce((a, c) => a + c.win, 0) / chunk.length);
    ws.push(chunk.length);
  }
  pav(ys, ws); // 非減少に整形
  return { x: xs, y: ys };
}

/** PAV: 重み付きで非減少になるよう y を書き換える。 */
function pav(y: number[], w: number[]): void {
  const val = [...y];
  const wt = [...w];
  const idx = y.map((_, i) => i);
  let i = 0;
  const blocks: { v: number; w: number; members: number[] }[] = val.map((v, k) => ({ v, w: wt[k]!, members: [k] }));
  const out: typeof blocks = [];
  for (const b of blocks) {
    out.push(b);
    while (out.length >= 2 && out[out.length - 2]!.v > out[out.length - 1]!.v) {
      const b2 = out.pop()!;
      const b1 = out.pop()!;
      const wsum = b1.w + b2.w;
      out.push({ v: (b1.v * b1.w + b2.v * b2.w) / wsum, w: wsum, members: [...b1.members, ...b2.members] });
    }
  }
  for (const b of out) for (const m of b.members) y[m] = b.v;
  void i; void idx;
}

/** 生確率 p を較正。x の外は端でクランプ、内は線形補間。 */
export function applyCalibration(p: number, cal: Calibration): number {
  const { x, y } = cal;
  if (x.length === 0) return p;
  if (p <= x[0]!) return y[0]!;
  if (p >= x[x.length - 1]!) return y[y.length - 1]!;
  for (let i = 1; i < x.length; i++) {
    if (p <= x[i]!) {
      const t = (p - x[i - 1]!) / (x[i]! - x[i - 1]! || 1);
      return y[i - 1]! + t * (y[i]! - y[i - 1]!);
    }
  }
  return y[y.length - 1]!;
}
