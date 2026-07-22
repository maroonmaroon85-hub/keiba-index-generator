/**
 * バックテストの集計。
 * 1サンプル = 1頭のあるレースでの (指数計算結果 + 実着順 + 確定オッズ)。
 */
export interface Sample {
  rank: string;
  mark: string;
  flags: string[]; // plus/minus をまとめたフラグ名
  winProb: number;
  placeProb: number;
  finish: number;
  fieldSize: number;
  /** 確定単勝オッズ（無ければ0）。 */
  winOdds: number;
}

export interface GroupStat {
  label: string;
  n: number;
  winRate: number; // 勝率(1着)
  quinellaRate: number; // 連対率(2着内)
  placeRate: number; // 複勝率(3着内)
  /** 単勝回収率(%)。オッズ欠損時は null。 */
  winROI: number | null;
}

function stat(label: string, samples: Sample[]): GroupStat {
  const n = samples.length;
  if (n === 0) return { label, n: 0, winRate: 0, quinellaRate: 0, placeRate: 0, winROI: null };
  const win = samples.filter((s) => s.finish === 1).length;
  const quin = samples.filter((s) => s.finish >= 1 && s.finish <= 2).length;
  const place = samples.filter((s) => s.finish >= 1 && s.finish <= 3).length;
  const withOdds = samples.filter((s) => s.winOdds > 0);
  let winROI: number | null = null;
  if (withOdds.length > 0) {
    const bet = withOdds.length * 100;
    const ret = withOdds.reduce((a, s) => a + (s.finish === 1 ? s.winOdds * 100 : 0), 0);
    winROI = (ret / bet) * 100;
  }
  return {
    label,
    n,
    winRate: (win / n) * 100,
    quinellaRate: (quin / n) * 100,
    placeRate: (place / n) * 100,
    winROI,
  };
}

export function byRank(samples: Sample[], order: string[]): GroupStat[] {
  return order.map((r) => stat(r, samples.filter((s) => s.rank === r)));
}

export function byMark(samples: Sample[], marks: string[]): GroupStat[] {
  return marks.map((m) => stat(m, samples.filter((s) => s.mark === m)));
}

export function byFlag(samples: Sample[]): GroupStat[] {
  const names = new Set<string>();
  for (const s of samples) for (const f of s.flags) names.add(f);
  const baseline = stat("(全体)", samples);
  const rows = [...names].sort().map((name) => stat(name, samples.filter((s) => s.flags.includes(name))));
  return [baseline, ...rows];
}

export interface EvBandStat {
  label: string;
  bets: number;
  hitRate: number; // 的中率(1着率)
  roi: number; // 単勝回収率(%)
}

/**
 * 単勝EV購入シミュレーション。
 * 各馬 EV = win_prob × 確定単勝オッズ。EV帯ごとに「その帯を全部100円買ったら」の的中率・回収率。
 * threshold 以上を買う戦略の総合回収率が 100% を超えるかが、システムの収益性の答え。
 */
export function evSimulation(samples: Sample[], bands: number[], contenderOnly = false): EvBandStat[] {
  const withOdds = samples.filter((s) => s.winOdds > 0 && (!contenderOnly || s.winProb >= 1 / s.fieldSize));
  const rows: EvBandStat[] = [];
  for (let i = 0; i < bands.length - 1; i++) {
    const lo = bands[i]!;
    const hi = bands[i + 1]!;
    const inBand = withOdds.filter((s) => {
      const ev = s.winProb * s.winOdds;
      return ev >= lo && ev < hi;
    });
    rows.push(evStat(`EV ${lo.toFixed(1)}-${hi.toFixed(1)}`, inBand));
  }
  return rows;
}

/** threshold 以上を全部買う戦略の総合成績。 */
export function evStrategy(samples: Sample[], threshold: number, contenderOnly = false): EvBandStat {
  const bets = samples.filter(
    (s) => s.winOdds > 0 && (!contenderOnly || s.winProb >= 1 / s.fieldSize) && s.winProb * s.winOdds >= threshold,
  );
  return evStat(`EV≥${threshold} 総合`, bets);
}

function evStat(label: string, bets: Sample[]): EvBandStat {
  const n = bets.length;
  if (n === 0) return { label, bets: 0, hitRate: 0, roi: 0 };
  const hits = bets.filter((s) => s.finish === 1).length;
  const ret = bets.reduce((a, s) => a + (s.finish === 1 ? s.winOdds * 100 : 0), 0);
  return { label, bets: n, hitRate: (hits / n) * 100, roi: (ret / (n * 100)) * 100 };
}

export interface CalibrationBin {
  label: string;
  n: number;
  predicted: number; // 予測平均(%)
  actualWin: number; // 実勝率(%)
  actualPlace: number; // 実複勝率(%)
}

/** win_prob 予測帯ごとの実勝率・実複勝率。 */
export function calibration(samples: Sample[], bins = [0, 5, 10, 20, 30, 50, 100]): CalibrationBin[] {
  const out: CalibrationBin[] = [];
  for (let i = 0; i < bins.length - 1; i++) {
    const lo = bins[i]! / 100;
    const hi = bins[i + 1]! / 100;
    const isLast = i === bins.length - 2;
    const inBin = samples.filter((s) => s.winProb >= lo && (isLast ? s.winProb <= hi : s.winProb < hi));
    const n = inBin.length;
    out.push({
      label: `${bins[i]}-${bins[i + 1]}%`,
      n,
      predicted: n ? (inBin.reduce((a, s) => a + s.winProb, 0) / n) * 100 : 0,
      actualWin: n ? (inBin.filter((s) => s.finish === 1).length / n) * 100 : 0,
      actualPlace: n ? (inBin.filter((s) => s.finish >= 1 && s.finish <= 3).length / n) * 100 : 0,
    });
  }
  return out;
}
