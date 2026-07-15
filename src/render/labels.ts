import type { PastRun, PreRaceHorse, Pace } from "../model/pre-race.js";
import type { ScoredHorse } from "../scoring/types.js";

/** 前走・二前・三前の着順評価マーク。1-3着=○, 4-5着=△, 6着以下=×, 圧勝=◎。B=ブリンカー着用走。 */
export function pastEvalMark(run: PastRun): { mark: "◎" | "○" | "△" | "×"; blinker: boolean } {
  let mark: "◎" | "○" | "△" | "×";
  // 圧勝: 1着かつ着差が大きい（margin>=0.4秒 ≒ 2馬身超）。
  if (run.finish === 1 && run.margin >= 0.4) mark = "◎";
  else if (run.finish <= 3) mark = "○";
  else if (run.finish <= 5) mark = "△";
  else mark = "×";
  return { mark, blinker: run.blinker };
}

/** 今走含む連続ブリンカー着用数（B{n}表示用）。今走非着用なら0。 */
export function blinkerStreak(horse: PreRaceHorse): number {
  if (!horse.blinker) return 0;
  let n = 1;
  for (const run of horse.pastRuns) {
    if (run.blinker) n++;
    else break;
  }
  return n;
}

/**
 * 性齢・ローテ ラベル。例: 牡3,中3-5-1,同2,B2
 *  - 中3 = 中3週
 *  - 5-1 = 二前5着→前走1着（古い→新しい）
 *  - 同2 = 同距離2走目（same 判定は呼び出し側の flag で渡す）
 *  - B2 = ブリンカー2走目
 */
export function rotationLabel(
  horse: PreRaceHorse,
  scored: ScoredHorse,
  raceDistance: number,
): string {
  const parts: string[] = [`${horse.sex}${horse.age}`];
  const finishes = horse.pastRuns
    .slice(0, 2)
    .reverse()
    .map((r) => r.finish);
  const rot = `中${horse.weeksSinceLastRun}${finishes.length ? "-" + finishes.join("-") : ""}`;
  parts.push(rot);
  const last = horse.pastRuns[0];
  if (last && last.distance === raceDistance) parts.push("同2");
  const bn = blinkerStreak(horse);
  if (bn >= 2) parts.push(`B${bn}`);
  return parts.join(",");
}

const STYLE_POS: Record<string, string> = { f: "先行", m: "中団", b: "後方" };

/** 予想位置: 脚質 × 枠 × 想定ペース。短いラベル。 */
export function predictedPosition(scored: ScoredHorse, frame: number, pace: Pace): string {
  const base = STYLE_POS[scored.style] ?? "中団";
  const side = frame <= 3 ? "内" : frame >= 7 ? "外" : "中";
  // ハイペースは差し・追込がやや前進、スローは先行有利（表示上の含み）。
  let hint = "";
  if (pace === "H" && scored.style === "b") hint = "△展開向";
  if (pace === "S" && scored.style === "f") hint = "△展開向";
  return `${base}(${side})${hint}`;
}

/** 減量記号込みの騎手表示。 */
export function jockeyLabel(horse: PreRaceHorse): string {
  return `${horse.jockey.reductionMark}${horse.jockey.name}`;
}

/** 調教トレンドの矢印。 */
export function trendArrow(trend: PreRaceHorse["training"]["trend"]): string {
  return trend === "up" ? "↗" : trend === "down" ? "↘" : "→";
}
