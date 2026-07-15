/**
 * 血統系統マスタ（仮）。
 *
 * 主要どころ約20系統の「系統名 → 表示色」と「系統 × コース種別 × 距離帯 → 適性(0..1)」。
 * 値はすべて仮置き。妥当性は Phase 3 のバックテストで検証・調整する前提なので凝らない。
 * ※ 実CSV接続(Phase 2)では 父馬名/母父馬名 → 系統名 の対応表を別途用意する想定。
 *   Phase 1 ではダミーデータが直接「系統名」を持つ。
 */

import type { Surface } from "../model/pre-race.js";

/** 距離帯。 */
export type DistanceBand = "sprint" | "mile" | "middle" | "long";

export function distanceBand(distance: number): DistanceBand {
  if (distance <= 1300) return "sprint";
  if (distance <= 1700) return "mile";
  if (distance <= 2200) return "middle";
  return "long";
}

/** 系統ごとの表示色（HTML 背景色）。 */
export const LINE_COLORS: Record<string, string> = {
  "サンデー系": "#ffd6e7",
  "ディープ系": "#ffc9de",
  "ハーツ系": "#f3c6ff",
  "キンカメ系": "#c9e2ff",
  "ロベルト系": "#d0bfff",
  "ミスプロ系": "#c5f6fa",
  "ヘイロー系": "#ffe8cc",
  "ノーザン系": "#d3f9d8",
  "ナスルーラ系": "#fff3bf",
  "グレイソヴリン系": "#e9d8a6",
  "ネイティヴ系": "#ffec99",
  "ダンチヒ系": "#a5d8ff",
  "ストームキャット系": "#99e9f2",
  "ヌレイエフ系": "#eebefa",
  "サドラー系": "#b2f2bb",
  "ニジンスキー系": "#c0eb75",
  "リファール系": "#d8f5a2",
  "フォルリ系": "#ffd8a8",
  "ファピアノ系": "#96f2d7",
  "その他": "#e9ecef",
};

/** 未知系統のフォールバック色。 */
export const DEFAULT_LINE_COLOR = "#e9ecef";

export function lineColor(line: string): string {
  return LINE_COLORS[line] ?? DEFAULT_LINE_COLOR;
}

/**
 * 適性テーブル: 系統 → コース種別 → 距離帯 → 0..1。
 * 定義がない組み合わせは 0.5（中立）にフォールバック。
 */
type Aptitude = Partial<Record<Surface, Partial<Record<DistanceBand, number>>>>;

const APTITUDE: Record<string, Aptitude> = {
  "サンデー系": { 芝: { sprint: 0.55, mile: 0.7, middle: 0.75, long: 0.65 }, ダ: { sprint: 0.5, mile: 0.5, middle: 0.45, long: 0.4 } },
  "ディープ系": { 芝: { sprint: 0.5, mile: 0.7, middle: 0.85, long: 0.8 }, ダ: { sprint: 0.4, mile: 0.45, middle: 0.4, long: 0.4 } },
  "ハーツ系": { 芝: { sprint: 0.45, mile: 0.6, middle: 0.8, long: 0.8 }, ダ: { sprint: 0.5, mile: 0.55, middle: 0.55, long: 0.5 } },
  "キンカメ系": { 芝: { sprint: 0.6, mile: 0.75, middle: 0.7, long: 0.6 }, ダ: { sprint: 0.65, mile: 0.7, middle: 0.6, long: 0.5 } },
  "ロベルト系": { 芝: { sprint: 0.45, mile: 0.55, middle: 0.7, long: 0.75 }, ダ: { sprint: 0.6, mile: 0.65, middle: 0.65, long: 0.6 } },
  "ミスプロ系": { 芝: { sprint: 0.6, mile: 0.65, middle: 0.55, long: 0.45 }, ダ: { sprint: 0.7, mile: 0.75, middle: 0.65, long: 0.55 } },
  "ヘイロー系": { 芝: { sprint: 0.65, mile: 0.6, middle: 0.5, long: 0.4 }, ダ: { sprint: 0.7, mile: 0.65, middle: 0.55, long: 0.45 } },
  "ノーザン系": { 芝: { sprint: 0.55, mile: 0.6, middle: 0.6, long: 0.6 }, ダ: { sprint: 0.6, mile: 0.6, middle: 0.6, long: 0.55 } },
  "ナスルーラ系": { 芝: { sprint: 0.6, mile: 0.6, middle: 0.55, long: 0.5 }, ダ: { sprint: 0.6, mile: 0.6, middle: 0.55, long: 0.5 } },
  "ダンチヒ系": { 芝: { sprint: 0.7, mile: 0.7, middle: 0.55, long: 0.4 }, ダ: { sprint: 0.65, mile: 0.65, middle: 0.5, long: 0.4 } },
  "ストームキャット系": { 芝: { sprint: 0.65, mile: 0.65, middle: 0.5, long: 0.4 }, ダ: { sprint: 0.75, mile: 0.7, middle: 0.55, long: 0.45 } },
  "ファピアノ系": { 芝: { sprint: 0.5, mile: 0.55, middle: 0.5, long: 0.45 }, ダ: { sprint: 0.75, mile: 0.75, middle: 0.7, long: 0.6 } },
};

/** 系統 × コース種別 × 距離帯 の適性(0..1)。未定義は 0.5。 */
export function aptitude(line: string, surface: Surface, distance: number): number {
  const band = distanceBand(distance);
  return APTITUDE[line]?.[surface]?.[band] ?? 0.5;
}
