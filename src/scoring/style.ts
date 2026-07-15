import type { PreRaceHorse, RunningStyle } from "../model/pre-race.js";

/**
 * 通過順位履歴から基本脚質を推定する。
 * 各近走の平均通過順位を「相対位置(0=先頭, 1=最後方)」に正規化し、
 * 全走の平均で f/m/b を決める。履歴が無ければ 'm'(差し) を既定とする。
 */
export function estimateStyle(horse: PreRaceHorse): RunningStyle {
  const positions: number[] = [];
  for (const run of horse.pastRuns) {
    if (run.passing.length === 0) continue;
    const avgPass = run.passing.reduce((a, b) => a + b, 0) / run.passing.length;
    // 相対位置: (通過順位-1)/(頭数-1)。頭数1は無視。
    if (run.fieldSize > 1) {
      positions.push((avgPass - 1) / (run.fieldSize - 1));
    }
  }
  if (positions.length === 0) return "m";
  const rel = positions.reduce((a, b) => a + b, 0) / positions.length;
  if (rel <= 0.33) return "f";
  if (rel <= 0.66) return "m";
  return "b";
}

const STYLE_LABEL: Record<RunningStyle, string> = {
  f: "逃げ先行",
  m: "差し",
  b: "追込",
};

export function styleLabel(style: RunningStyle): string {
  return STYLE_LABEL[style];
}
