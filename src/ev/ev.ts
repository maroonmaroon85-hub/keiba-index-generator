import type { ScoredRace, ScoredHorse } from "../scoring/types.js";
import type { Config } from "../config.js";

/**
 * 期待値(EV)による買い目算出。
 * EV = 想定確率 × オッズ。EV >= threshold を「買い」とする。
 *  - 単勝: win_prob × 単勝オッズ
 *  - 複勝: place_prob × 複勝オッズ（複勝オッズが与えられた場合のみ）
 *
 * 想定確率(win_prob/place_prob)は Phase 3 でキャリブレーション済み。
 * オッズは発走前オッズ（出馬表CSVの単勝、複勝は手入力/別CSV）を渡す。
 */

export interface OddsInput {
  /** 馬番 → 単勝オッズ。 */
  win: Map<number, number>;
  /** 馬番 → 複勝オッズ（下限側。無ければ空）。 */
  place?: Map<number, number>;
}

export interface Bet {
  number: number;
  name: string;
  type: "単勝" | "複勝";
  prob: number;
  odds: number;
  ev: number;
}

/** レースのスコア＋オッズ → EVが閾値以上の買い目一覧（EV降順）。 */
export function buyList(scored: ScoredRace, odds: OddsInput, config: Config): Bet[] {
  const bets: Bet[] = [];
  // contenderOnly: モデルが「平均以上」と見た馬のみ対象（win_prob >= 1/頭数）。
  // 人気薄×高オッズでEVだけ跳ねる馬（実は勝てない）を除外する。
  const floor = config.ev.contenderOnly ? 1 / scored.horses.length : 0;
  for (const h of scored.horses) {
    if (h.winProb < floor) continue;
    const w = odds.win.get(h.number);
    if (w && w > 0) {
      const ev = h.winProb * w;
      if (ev >= config.ev.threshold) {
        bets.push({ number: h.number, name: h.name, type: "単勝", prob: h.winProb, odds: w, ev: round2(ev) });
      }
    }
    const p = odds.place?.get(h.number);
    if (p && p > 0) {
      const ev = h.placeProb * p;
      if (ev >= config.ev.threshold) {
        bets.push({ number: h.number, name: h.name, type: "複勝", prob: h.placeProb, odds: p, ev: round2(ev) });
      }
    }
  }
  return bets.sort((a, b) => b.ev - a.ev);
}

/** 出馬表CSVの単勝オッズ（PreRaceHorse.odds）から単勝オッズマップを作る。 */
export function winOddsFromHorses(horses: { number: number; odds: number }[]): OddsInput {
  const win = new Map<number, number>();
  for (const h of horses) if (h.odds > 0) win.set(h.number, h.odds);
  return { win };
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}
