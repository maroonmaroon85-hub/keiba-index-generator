/**
 * PostRaceData — レース後に確定する情報。
 *
 * 【重要】これは Phase 3 バックテストの「答え合わせ」専用。
 * scoring / rules へは絶対に渡さない（型でも渡せないようにシグネチャを分離している）。
 * Phase 5(ML化) では、この型が学習ラベルの供給源になる。
 */

export interface PostRaceHorse {
  /** 馬番（PreRaceHorse.number と対応）。 */
  number: number;
  /** 確定着順。 */
  finish: number;
  /** 確定単勝オッズ。 */
  finalWinOdds: number;
  /** 単勝払戻（100円あたり円。非的中は0）。 */
  winPayout: number;
  /** 複勝払戻（100円あたり円。非的中は0）。 */
  placePayout: number;
}

export interface PostRaceData {
  raceId: string;
  horses: PostRaceHorse[];
}
