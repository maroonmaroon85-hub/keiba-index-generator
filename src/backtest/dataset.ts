import { readCsvShiftJis, toInt, toNum } from "../parser/csv.js";
import { sireToLine } from "../pedigree/sire-lines.js";
import type {
  PreRaceData,
  PreRaceHorse,
  PastRun,
  Surface,
  TrackCondition,
} from "../model/pre-race.js";
import type { PostRaceData, PostRaceHorse } from "../model/post-race.js";

/**
 * バックテスト用データセット構築。
 *
 * 入力: TARGET「全馬全成績 → 成績フルセット（＋単オッズ）」を期間分まとめた1CSV（Shift_JIS・52列以上）。
 * 1行 = ある馬のあるレースでの結果。これを:
 *  - レースID(col41)でグループ化 → 各レースのフルフィールド（PostRaceの答え）
 *  - 各馬の「そのレースより前の行」から近走特徴量を再構成（PreRaceData）
 * にして、指数計算部（PreRaceのみ）と答え合わせ（PostRace）を分離したまま流せるようにする。
 *
 * 【リーク防止】PreRaceHorse には当該レースの着順・オッズを一切入れない。
 *  近走(pastRuns)は「当該レースより前」の行のみ。PostRaceData に着順・確定オッズを分離。
 */

interface Row {
  raceId: string;
  date: Date;
  course: string;
  surface: Surface;
  distance: number;
  condition: TrackCondition;
  raceName: string;
  fieldSize: number;
  umaban: number;
  horseKey: string; // 血統登録番号（無ければ馬名）
  horseName: string;
  sex: string;
  age: number;
  weightCarry: number;
  finish: number;
  margin: number;
  passing: number[];
  sireLine: string;
  damSireLine: string;
  producer: string;
  /** 単オッズ（フルセット+単オッズ export のときのみ。--odds-col 指定）。 */
  winOdds?: number;
}

export interface BacktestRace {
  pre: PreRaceData;
  post: PostRaceData;
}

function mapCondition(v: string): TrackCondition {
  const c = (v ?? "").trim().charAt(0);
  if (c === "稍") return "稍";
  if (c === "重") return "重";
  if (c === "不") return "不";
  return "良";
}

function parseRow(r: string[], oddsCol?: number): Row | null {
  if (r.length < 48) return null;
  const finish = toInt(r[20]);
  const fieldSize = toInt(r[18]);
  // col41 = レースID + 馬番（末尾2桁が馬番）。フィールド復元のため馬番を切り離す。
  const rawKey = (r[40] ?? "").trim();
  const raceId = rawKey.length > 2 ? rawKey.slice(0, -2) : rawKey;
  const umaban = rawKey.length > 2 ? toInt(rawKey.slice(-2)) : 0;
  const horseName = (r[13] ?? "").trim();
  if (!raceId || !horseName) return null;
  const key = (r[37] ?? "").trim() || horseName;
  const passing = [r[28], r[29], r[30], r[31]]
    .map((x) => toInt(x))
    .filter((n) => n > 0);
  return {
    raceId,
    date: new Date(2000 + toInt(r[0]), toInt(r[1]) - 1, toInt(r[2])),
    course: (r[4] ?? "").trim(),
    surface: (r[9] === "芝" ? "芝" : "ダ") as Surface,
    distance: toInt(r[11]),
    condition: mapCondition(r[12] ?? ""),
    raceName: (r[7] ?? "").trim(),
    fieldSize,
    umaban,
    horseKey: key,
    horseName,
    sex: (r[14] ?? "").trim(),
    age: toInt(r[15]),
    weightCarry: toNum(r[17]),
    finish,
    margin: toNum(r[23]),
    passing,
    sireLine: sireToLine(r[43] ?? ""),
    damSireLine: sireToLine(r[45] ?? ""),
    producer: (r[42] ?? "").trim(),
    ...(oddsCol !== undefined ? { winOdds: toNum(r[oddsCol]) } : {}),
  };
}

/** Row → 近走特徴量(PreRaceHorse相当)。priorRuns は当該レースより前（新しい順）。 */
function buildHorse(row: Row, priorRuns: Row[]): PreRaceHorse {
  const pastRuns: PastRun[] = priorRuns.slice(0, 3).map((p) => ({
    finish: p.finish,
    fieldSize: p.fieldSize,
    surface: p.surface,
    distance: p.distance,
    course: p.course,
    passing: p.passing,
    margin: p.margin,
    blinker: false,
  }));
  const weeks = (a: Date, b: Date) =>
    Math.max(0, Math.round((a.getTime() - b.getTime()) / (7 * 864e5)));
  const last = priorRuns[0];
  const prev = priorRuns[1];
  return {
    number: 0, // 馬番は成績CSVに無い→0（内枠系ルールは評価対象外）
    frame: 0,
    name: row.horseName,
    blinker: false,
    jockey: { name: "", reductionMark: "", reductionKg: 0 },
    sex: row.sex === "牝" ? "牝" : row.sex === "セ" ? "セ" : "牡",
    age: row.age,
    weightCarry: row.weightCarry,
    prevWeightCarry: last ? last.weightCarry : row.weightCarry,
    odds: row.winOdds ?? 0,
    pastRuns,
    weeksSinceLastRun: last ? weeks(row.date, last.date) : 99,
    weeksBeforeLastRun: last && prev ? weeks(last.date, prev.date) : 99,
    sireLine: row.sireLine,
    damSireLine: row.damSireLine,
    training: { evaluation: "B", trend: "flat" },
    producer: row.producer,
    trainer: "",
  };
}

/**
 * 成績CSV → バックテスト用レース群。
 * minHorses 未満の（フィールドが揃わない）レースは除外。
 */
export function buildDataset(path: string, opts: { oddsCol?: number; minHorses?: number } = {}): BacktestRace[] {
  const minHorses = opts.minHorses ?? 5;
  const rows = readCsvShiftJis(path)
    .map((r) => parseRow(r, opts.oddsCol))
    .filter((x): x is Row => x !== null);

  // 馬ごとに時系列（昇順）で索引。
  const byHorse = new Map<string, Row[]>();
  for (const row of rows) {
    (byHorse.get(row.horseKey) ?? byHorse.set(row.horseKey, []).get(row.horseKey)!).push(row);
  }
  for (const list of byHorse.values()) list.sort((a, b) => a.date.getTime() - b.date.getTime());

  // レースごとにグループ化。
  const byRace = new Map<string, Row[]>();
  for (const row of rows) {
    (byRace.get(row.raceId) ?? byRace.set(row.raceId, []).get(row.raceId)!).push(row);
  }

  const races: BacktestRace[] = [];
  for (const [raceId, field] of byRace) {
    if (field.length < minHorses) continue;
    const head = field[0]!;
    const horses: PreRaceHorse[] = [];
    const post: PostRaceHorse[] = [];
    field.forEach((row, i) => {
      const hist = byHorse.get(row.horseKey) ?? [];
      const prior = hist.filter((h) => h.date.getTime() < row.date.getTime()).reverse(); // 新しい順
      const horse = buildHorse(row, prior);
      const num = row.umaban || i + 1;
      horse.number = num;
      horses.push(horse);
      post.push({
        number: num,
        finish: row.finish,
        finalWinOdds: row.winOdds ?? 0,
        winPayout: row.winOdds && row.finish === 1 ? Math.round(row.winOdds * 100) : 0,
        placePayout: 0, // 複勝払戻は成績CSVに無い（別途取得時に対応）
      });
    });
    races.push({
      pre: {
        race: {
          raceId,
          course: head.course,
          surface: head.surface,
          distance: head.distance,
          pace: "M", // 過去のペースは不明→中立
          condition: head.condition,
          name: head.raceName,
        },
        horses,
      },
      post: { raceId, horses: post },
    });
  }
  return races;
}
