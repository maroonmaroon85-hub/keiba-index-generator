import type { RacePayout, Winning } from "./exotic-sim.js";

/**
 * JRA公式「レース結果」ページのテキストから払戻を抽出する。
 * ユーザーが結果ページを貼る運用を想定（この環境はネット制限でJRAに直接アクセス不可）。
 * 払戻節の形式（タブ/改行区切り）:
 *   単勝 \n 1 \t 1,780円 \t 4番人気
 *   複勝 \n 1 \t 130円 ... \n 3 \t 100円 ...
 *   ワイド \n 1-3 \t 190円 ... （3組）
 *   馬連 \n 1-3 \t 590円
 *   馬単 \n 1-3 \t 2,840円  （着順どおり）
 *   3連複 \n 1-3-5 \t 470円
 *   3連単 \n 1-3-5 \t 8,490円 （着順どおり）
 */

export interface ParsedResult {
  date: string; // 例 2026-07-04（抽出できなければ空）
  course: string; // 例 福島
  raceNo: number; // 例 1
  payout: RacePayout;
}

const TYPES = ["単勝", "複勝", "枠連", "ワイド", "馬連", "馬単", "3連複", "3連単", "三連複", "三連単"];

function yen(s: string): number {
  const m = s.replace(/,/g, "").match(/(\d+)\s*円/);
  return m ? parseInt(m[1]!, 10) : 0;
}
function combo(s: string): number[] {
  return s.trim().split("-").map((x) => parseInt(x, 10)).filter((n) => Number.isFinite(n));
}

/** JRA結果ページのテキスト全体 → 払戻。 */
export function parseJraResult(text: string): ParsedResult {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter((l) => l !== "");

  // レースメタ（例: 「2026年7月4日（土曜） 2回福島3日」「1レース」）。
  let date = "";
  let course = "";
  let raceNo = 0;
  for (const l of lines) {
    const dm = l.match(/(\d{4})年(\d{1,2})月(\d{1,2})日/);
    if (dm && !date) date = `${dm[1]}-${dm[2]!.padStart(2, "0")}-${dm[3]!.padStart(2, "0")}`;
    const cm = l.match(/\d回(\D{2,3}?)\d+日/);
    if (cm && !course) course = cm[1]!;
    const rm = l.match(/^(\d{1,2})\s*レース$/);
    if (rm && !raceNo) raceNo = parseInt(rm[1]!, 10);
  }

  const payout: RacePayout = {};
  const wide: Winning[] = [];
  let cur = "";
  for (const l of lines) {
    if (TYPES.includes(l)) {
      cur = l;
      continue;
    }
    if (!cur) continue;
    // データ行: 「1-3-5 470円 2番人気」等。組番トークン＋金額。
    const parts = l.split(/[\t\s]+/).filter((x) => x !== "");
    if (parts.length < 2) continue;
    if (!/^[\d-]+$/.test(parts[0]!)) continue;
    const c = combo(parts[0]!);
    const amt = yen(l);
    if (amt === 0 || c.length === 0) continue;
    const w: Winning = { combo: c, payout: amt };
    switch (cur) {
      case "馬連": payout.umaren = w; break;
      case "馬単": payout.umatan = w; break;
      case "3連複": case "三連複": payout.sanrenpuku = w; break;
      case "3連単": case "三連単": payout.sanrentan = w; break;
      case "ワイド": wide.push(w); break;
      // 単勝・複勝・枠連は連系バックテストでは未使用（必要なら拡張）。
    }
  }
  if (wide.length) payout.wide = wide;
  return { date, course, raceNo, payout };
}
