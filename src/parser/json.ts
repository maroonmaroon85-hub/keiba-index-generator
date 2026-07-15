import { readFileSync } from "node:fs";
import type { PreRaceData, Pace, TrackCondition } from "../model/pre-race.js";

/**
 * Phase 1 用ローダー。ダミーJSON（内部モデル PreRaceData と同形）を読み込む。
 * Phase 2 で TARGET の実CSV→PreRaceData 変換に置き換わるが、
 * scoring 以降のパイプラインは同じ PreRaceData を受け取る。
 */
export function loadPreRaceFromJson(path: string): PreRaceData {
  let raw: string;
  try {
    raw = readFileSync(path, "utf-8");
  } catch {
    throw new Error(`入力ファイルを読めません: ${path}`);
  }
  let data: PreRaceData;
  try {
    data = JSON.parse(raw) as PreRaceData;
  } catch (e) {
    throw new Error(`入力JSONの構文エラー: ${path}\n${(e as Error).message}`);
  }
  validate(data, path);
  return data;
}

function validate(data: PreRaceData, path: string): void {
  if (!data.race) throw new Error(`race がありません: ${path}`);
  if (!Array.isArray(data.horses) || data.horses.length === 0) {
    throw new Error(`horses が空です: ${path}`);
  }
  const numbers = new Set<number>();
  for (const h of data.horses) {
    if (typeof h.number !== "number") {
      throw new Error(`馬番(number)が不正な馬があります: ${path} (${h.name ?? "?"})`);
    }
    if (numbers.has(h.number)) {
      throw new Error(`馬番が重複しています: ${h.number} (${path})`);
    }
    numbers.add(h.number);
    if (!Array.isArray(h.pastRuns)) {
      throw new Error(`pastRuns が配列でない馬があります: ${path} (馬番 ${h.number})`);
    }
  }
}

/** CLI引数で pace / condition を上書き（手入力の割り切り）。 */
export function overrideRaceParams(
  data: PreRaceData,
  pace?: Pace,
  condition?: TrackCondition,
): PreRaceData {
  if (pace) data.race.pace = pace;
  if (condition) data.race.condition = condition;
  return data;
}
