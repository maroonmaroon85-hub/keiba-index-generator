import type { PreRaceHorse, PreRaceInfo } from "../model/pre-race.js";

/**
 * 条件フラグ（ルール）の定義。1ルール1ファイル。
 * name は config.rules / 表示ラベルと一致させる（例: "延長△"）。
 * weight は config.json 側で管理し、scoring 時に name で引く。
 *
 * 【リーク防止】condition は PreRaceHorse / PreRaceInfo しか受け取れない。
 */
export interface RuleContext {
  /** config.ruleParams。 */
  params: Record<string, number>;
}

export interface Rule {
  /** config.rules のキー兼 表示ラベル（末尾に ◎/△ を含む）。 */
  name: string;
  sign: "plus" | "minus";
  condition: (horse: PreRaceHorse, race: PreRaceInfo, ctx: RuleContext) => boolean;
}
