import { readCsvShiftJis, toInt, toNum } from "./csv.js";
import type {
  PreRaceData,
  PreRaceHorse,
  PreRaceInfo,
  PastRun,
  Surface,
  Sex,
} from "../model/pre-race.js";
import { sireToLine } from "../pedigree/sire-lines.js";

/**
 * TARGET frontier JV の実CSV（Shift_JIS）→ 内部モデル PreRaceData。
 * 3段構成: ①CSV読み込み ②馬名で結合 ③PreRaceData構築。
 * 列対応は data/sample/schema.md を参照。
 *
 * 【リーク防止】ここで作るのは PreRaceData のみ。着順等の確定情報(PostRace)は扱わない。
 * ※ seiseki は「過去走」の着順であり、当該レースの結果ではないので PreRace 情報として正当。
 */

/** CLI から渡すレースヘッダ（CSVにヘッダ行が無いため）。 */
export interface TargetRaceHeader {
  raceId: string;
  course: string;
  surface: Surface;
  distance: number;
  pace: PreRaceInfo["pace"];
  condition: PreRaceInfo["condition"];
  /** 当該レース発走日 YYYY-MM-DD。ローテ間隔の算出に使用。 */
  date: string;
  name?: string;
}

interface SeisekiHorse {
  pastRuns: PastRun[];
  sireLine: string;
  damSireLine: string;
  producer: string;
  bloodRegNo: string;
  /** 最新走の斤量（増量△判定用の prevWeightCarry）。 */
  prevWeightCarry: number;
  /** 走破日（新しい順）。ローテ算出用。 */
  runDates: Date[];
}

/** seiseki を馬名でグループ化して各馬の近走・血統を作る。 */
function parseSeiseki(rows: string[][]): Map<string, SeisekiHorse> {
  const byName = new Map<string, string[][]>();
  for (const r of rows) {
    if (r.length < 48) continue; // 不完全行はスキップ
    const name = (r[13] ?? "").trim();
    if (!name) continue;
    (byName.get(name) ?? byName.set(name, []).get(name)!).push(r);
  }

  const result = new Map<string, SeisekiHorse>();
  for (const [name, list] of byName) {
    // seiseki は日付降順。先頭が最新走。
    const pastRuns: PastRun[] = [];
    const runDates: Date[] = [];
    for (const r of list) {
      const surface = (r[9] === "芝" ? "芝" : "ダ") as Surface;
      const passing = [r[28], r[29], r[30], r[31]]
        .map((x) => parseInt(x ?? "", 10))
        .filter((n) => Number.isFinite(n) && n > 0); // 0（未計測）は除外
      pastRuns.push({
        finish: toInt(r[20]),
        fieldSize: toInt(r[18]),
        surface,
        distance: toInt(r[11]),
        course: (r[4] ?? "").trim(),
        passing,
        margin: toNum(r[23]),
        blinker: false, // seisekiにブリンカー着用有無は無いため既定false
      });
      runDates.push(toDate(r[0], r[1], r[2]));
    }
    const head = list[0]!;
    result.set(name, {
      pastRuns,
      sireLine: sireToLine(head[43] ?? ""),
      damSireLine: sireToLine(head[45] ?? ""),
      producer: (head[42] ?? "").trim(),
      bloodRegNo: (head[37] ?? "").trim(),
      prevWeightCarry: toNum(head[17]),
      runDates,
    });
  }
  return result;
}

const REDUCTION: Record<string, number> = { "☆": 1, "△": 2, "▲": 3, "★": 1 };

/** shutuba2（出馬表・画面イメージCSV）＋ seiseki を結合して PreRaceData を作る。 */
export function parseTargetCsv(
  shutubaPath: string,
  seisekiPath: string,
  header: TargetRaceHeader,
): PreRaceData {
  const shutubaRows = readCsvShiftJis(shutubaPath);
  const seisekiRows = readCsvShiftJis(seisekiPath);
  if (shutubaRows.length < 2) {
    throw new Error(`出馬表CSVに馬データがありません: ${shutubaPath}`);
  }
  const seiseki = parseSeiseki(seisekiRows);
  const raceDate = new Date(header.date);
  if (Number.isNaN(raceDate.getTime())) {
    throw new Error(`--date が不正です（YYYY-MM-DD）: ${header.date}`);
  }

  const horses: PreRaceHorse[] = [];
  // 1行目はヘッダ（枠番,B,番,...）なので2行目以降。
  for (let i = 1; i < shutubaRows.length; i++) {
    const r = shutubaRows[i]!;
    if (r.length < 24) continue;
    const name = (r[7] ?? "").trim();
    if (!name) continue;
    const s = seiseki.get(name);
    const reductionMark = (r[14] ?? "").trim();

    const weeksSinceLastRun = s ? weeksBetween(s.runDates[0], raceDate) : 99;
    const weeksBeforeLastRun =
      s && s.runDates.length >= 2 ? weeksBetween(s.runDates[1], s.runDates[0]) : 99;

    horses.push({
      number: toInt(r[2]),
      frame: toInt(r[0]),
      name,
      blinker: (r[1] ?? "").trim() === "B",
      jockey: {
        name: (r[12] ?? "").trim(),
        reductionMark: (reductionMark as PreRaceHorse["jockey"]["reductionMark"]) || "",
        reductionKg: REDUCTION[reductionMark] ?? 0,
      },
      sex: normalizeSex(r[9]),
      age: toInt(r[10]),
      weightCarry: toNum(r[13]),
      prevWeightCarry: s ? s.prevWeightCarry : toNum(r[13]),
      odds: toNum(r[15]),
      pastRuns: s ? s.pastRuns : [],
      weeksSinceLastRun,
      weeksBeforeLastRun,
      sireLine: s ? s.sireLine : "その他",
      damSireLine: s ? s.damSireLine : "その他",
      training: { evaluation: "B", trend: "flat" }, // 調教CSV未取得→中立
      producer: s?.producer || (r[22] ?? "").trim(),
      trainer: (r[19] ?? "").trim(),
    });
  }

  if (horses.length === 0) {
    throw new Error(`出馬表CSVから馬を1頭も読めませんでした: ${shutubaPath}`);
  }

  const race: PreRaceInfo = {
    raceId: header.raceId,
    course: header.course,
    surface: header.surface,
    distance: header.distance,
    pace: header.pace,
    condition: header.condition,
    ...(header.name ? { name: header.name } : {}),
  };
  return { race, horses };
}

function normalizeSex(v: string | undefined): Sex {
  const s = (v ?? "").trim();
  if (s === "牝") return "牝";
  if (s === "セ") return "セ";
  return "牡";
}

/** seiseki の 年,月,日（2桁年）→ Date。 */
function toDate(y: string | undefined, m: string | undefined, d: string | undefined): Date {
  const year = 2000 + toInt(y);
  return new Date(year, toInt(m) - 1, toInt(d));
}
function weeksBetween(from: Date | undefined, to: Date | undefined): number {
  if (!from || !to) return 99;
  const ms = to.getTime() - from.getTime();
  return Math.max(0, Math.round(ms / (7 * 24 * 3600 * 1000)));
}
