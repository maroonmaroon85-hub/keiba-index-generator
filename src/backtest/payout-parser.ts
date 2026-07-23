import { readCsvShiftJis, toInt } from "../parser/csv.js";
import type { RacePayout, Winning } from "./exotic-sim.js";

/**
 * TARGET 払戻CSV（配当A / 配当B, JV-Data HRレコード相当・Shift_JIS・固定列）→ Map<raceId, RacePayout>。
 *
 * 実データから確定した列レイアウト（ヘッダ0-14, col14=raceId=成績col41先頭8桁と一致）:
 *
 * 配当B（136列, 2013-2026フル）: 単複枠馬連ワイドのみ。3連系なし。
 *   単勝 15  複勝 21  枠連 31  馬連 40  ワイド 49（以降オッズ表, 使わない）
 *   2頭系スロット=(番,配当), 枠/馬連/ワイド=(番,番,配当)。
 *
 * 配当A（224列, サンプル期間）: オッズ表(15-86)の後にフル券種（人気付き）。
 *   単勝 87  複勝 93  枠連 103  馬連 115  ワイド 127  馬単 155  三連複 179  三連単 194
 *   単複=(番,配当), 枠/馬連/ワイド/馬単=(番,番,配当,人気), 三連複/三連単=(番,番,番,配当,人気)。
 *
 * 使うのは馬連/ワイド/馬単/三連複/三連単のみ（研究対象）。単勝/複勝/枠連は読まない。
 */

const RACE_ID = 14;

/** payout>0 の的中スロットだけを返す（padの0スロットは捨てる）。 */
function readSlots(
  r: string[],
  start: number,
  slots: number,
  width: number,
  nHorses: number,
  payIdx: number,
): Winning[] {
  const out: Winning[] = [];
  for (let s = 0; s < slots; s++) {
    const base = start + s * width;
    const payout = toInt(r[base + payIdx]);
    if (payout <= 0) continue;
    const combo: number[] = [];
    let ok = true;
    for (let h = 0; h < nHorses; h++) {
      const n = toInt(r[base + h]);
      if (n <= 0) {
        ok = false;
        break;
      }
      combo.push(n);
    }
    if (ok) out.push({ combo, payout });
  }
  return out;
}

/** 配当B（136列）を Map に読み込む。馬連/ワイドのみ抽出。 */
export function parsePayoutB(path: string): Map<string, RacePayout> {
  const rows = readCsvShiftJis(path);
  const map = new Map<string, RacePayout>();
  for (const r of rows) {
    if (r.length < 58) continue;
    const raceId = (r[RACE_ID] ?? "").trim();
    if (!raceId) continue;
    // 馬連: start 40, 3 slots, width 3, 2頭, payIdx 2
    const umaren = readSlots(r, 40, 3, 3, 2, 2);
    // ワイド: start 49, 5 slots(pad込), width 3, 2頭, payIdx 2
    const wide = readSlots(r, 49, 5, 3, 2, 2);
    const p: RacePayout = {};
    if (umaren[0]) p.umaren = umaren[0];
    if (wide.length) p.wide = wide;
    map.set(raceId, p);
  }
  return map;
}

/** 配当A（224列）を Map に読み込む。馬連/ワイド/馬単/三連複/三連単を抽出。 */
export function parsePayoutA(path: string): Map<string, RacePayout> {
  const rows = readCsvShiftJis(path);
  const map = new Map<string, RacePayout>();
  for (const r of rows) {
    if (r.length < 224) continue;
    const raceId = (r[RACE_ID] ?? "").trim();
    if (!raceId) continue;
    // 馬連 115(3×4,payIdx2) ワイド 127(7×4) 馬単 155(6×4) 三連複 179(3×5,payIdx3) 三連単 194(6×5)
    const umaren = readSlots(r, 115, 3, 4, 2, 2);
    const wide = readSlots(r, 127, 7, 4, 2, 2);
    const umatan = readSlots(r, 155, 6, 4, 2, 2);
    const sanpuku = readSlots(r, 179, 3, 5, 3, 3);
    const santan = readSlots(r, 194, 6, 5, 3, 3);
    const p: RacePayout = {};
    if (umaren[0]) p.umaren = umaren[0];
    if (wide.length) p.wide = wide;
    if (umatan[0]) p.umatan = umatan[0];
    if (sanpuku[0]) p.sanrenpuku = sanpuku[0];
    if (santan[0]) p.sanrentan = santan[0];
    map.set(raceId, p);
  }
  return map;
}

/**
 * A・Bを両方読み、マージした Map を返す。
 * B（13年・馬連/ワイド）を土台に、A（3連系含む）を上書き・追加する。
 * 同一raceIdの馬連/ワイドはA・Bで一致するはずなのでAを優先しても問題ない。
 */
export function loadPayouts(pathB?: string, pathA?: string): Map<string, RacePayout> {
  const map = pathB ? parsePayoutB(pathB) : new Map<string, RacePayout>();
  if (pathA) {
    for (const [raceId, pa] of parsePayoutA(pathA)) {
      map.set(raceId, { ...map.get(raceId), ...pa });
    }
  }
  return map;
}
