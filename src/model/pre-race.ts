/**
 * PreRaceData — 発走前に見える情報のみ。
 *
 * 【大原則・リーク防止】
 * scoring / rules はこの PreRaceData しか受け取れないシグネチャにする。
 * 当該レースの着順・確定オッズ・払戻などの PostRaceData は絶対に混ぜない。
 * この型分離は Phase 5(ML化) でも特徴量/ラベル分離としてそのまま使う前提。
 */

export type Surface = "芝" | "ダ";
export type Sex = "牡" | "牝" | "セ";
/** 想定ペース。自動推定せずレース単位で手入力（CLI引数 or 設定）。 */
export type Pace = "S" | "M" | "H";
/** 馬場状態。自動推定せず手入力。 */
export type TrackCondition = "良" | "稍" | "重" | "不";

/** 基本脚質。通過順位履歴から推定。f=逃げ先行, m=差し, b=追込。 */
export type RunningStyle = "f" | "m" | "b";

export type TrainingEvaluation = "A" | "B" | "C";
export type TrainingTrend = "up" | "flat" | "down";

/** 騎手の減量記号。☆=1kg, △=2kg, ▲=3kg(女性), ★=見習い等。kg は斤量減の実数。 */
export interface Jockey {
  name: string;
  /** 減量記号（なければ空文字）。表示用。 */
  reductionMark: "" | "☆" | "△" | "▲" | "★";
  /** 減量kg（0=減量なし）。減量◎ルールで使用。 */
  reductionKg: number;
}

/** 近走1走分の成績（発走前に見えるので PreRace 扱い）。 */
export interface PastRun {
  /** 着順。 */
  finish: number;
  /** そのレースの出走頭数。 */
  fieldSize: number;
  surface: Surface;
  distance: number;
  course: string;
  /** 通過順位（例: [3,3,2]）。脚質推定に使用。 */
  passing: number[];
  /** 着差（勝ち馬とのタイム差・馬身換算の秒。圧勝◎判定用。勝利時は下位との差を負で表現しない簡易値）。 */
  margin: number;
  /** ブリンカー着用走か。 */
  blinker: boolean;
}

export interface PreRaceHorse {
  /** 馬番。 */
  number: number;
  /** 枠番。内枠系ルールで使用。 */
  frame: number;
  name: string;
  /** ブリンカー[B]着用（今走）。 */
  blinker: boolean;
  jockey: Jockey;
  sex: Sex;
  age: number;
  /** 斤量(kg)。増量△ルールで前走比を見る。 */
  weightCarry: number;
  /** 前走の斤量(kg)。 */
  prevWeightCarry: number;
  /** 前日/当日想定オッズ（単勝）。予想人気の算出元。 */
  odds: number;
  /** 近走成績（直近→過去の順で最大3走）。 */
  pastRuns: PastRun[];
  /** 前走からの間隔（週）。ローテ・叩き系ルールで使用。 */
  weeksSinceLastRun: number;
  /** 前走の前の間隔（週）。叩き2◎（休み明け2走目）判定に使用。 */
  weeksBeforeLastRun: number;
  /** 父系（系統名）。血統適性・表示色で使用。 */
  sireLine: string;
  /** 母父系（系統名）。 */
  damSireLine: string;
  training: {
    evaluation: TrainingEvaluation;
    trend: TrainingTrend;
  };
  /** 生産地。 */
  producer: string;
  /** 調教師。 */
  trainer: string;
}

export interface PreRaceInfo {
  /** レースID（例: 202607040611）。 */
  raceId: string;
  course: string;
  surface: Surface;
  distance: number;
  /** 手入力の想定ペース。 */
  pace: Pace;
  /** 手入力の馬場状態。 */
  condition: TrackCondition;
  /** レース名（任意）。 */
  name?: string;
}

export interface PreRaceData {
  race: PreRaceInfo;
  horses: PreRaceHorse[];
}
