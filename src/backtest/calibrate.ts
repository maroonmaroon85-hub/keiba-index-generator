import { writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { loadConfig } from "../config.js";
import { scoreRace } from "../scoring/score.js";
import { buildDataset, type BacktestRace } from "./dataset.js";
import { fitIsotonic, applyCalibration, type Calibration } from "./isotonic.js";

/**
 * 単調較正の学習＆アウトオブサンプル評価。
 * 期間の前半で較正を学習し、後半でEV回収率を「生確率 vs 較正確率」で比較する。
 * リーク防止のため学習期間と評価期間を日付で分離する。
 */

interface Args {
  input: string[];
  oddsCol?: number;
  minHorses?: number;
  splitRatio: number; // 前半何割を学習に使うか
  outDir: string;
  config?: string;
}

function parseArgs(argv: string[]): Args {
  const args: Args = { input: [], splitRatio: 0.7, outDir: "out" };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = argv[i + 1];
    switch (a) {
      case "--input": if (next) args.input.push(next); i++; break;
      case "--odds-col": args.oddsCol = Number(next) - 1; i++; break;
      case "--min-horses": args.minHorses = Number(next); i++; break;
      case "--split": args.splitRatio = Number(next); i++; break;
      case "--out": args.outDir = next ?? "out"; i++; break;
      case "--config": args.config = next; i++; break;
    }
  }
  return args;
}

interface Scored {
  rawProb: number;
  odds: number;
  finish: number;
  fieldSize: number;
  raceIdx: number; // 同一レースの再正規化用
}

/** レース群をスコアして (生確率, オッズ, 着順) サンプルへ。 */
function scoreAll(races: BacktestRace[], config: ReturnType<typeof loadConfig>): Scored[][] {
  return races.map((r) => {
    const scored = scoreRace(r.pre, config);
    const finishByNum = new Map(r.post.horses.map((h) => [h.number, h]));
    const out: Scored[] = [];
    for (const h of scored.horses) {
      const p = finishByNum.get(h.number);
      if (!p || p.finish <= 0) continue;
      out.push({ rawProb: h.winProb, odds: p.finalWinOdds, finish: p.finish, fieldSize: scored.horses.length, raceIdx: 0 });
    }
    return out;
  });
}

/** EV戦略の単勝回収率(%)。calibration があれば較正確率を各レースで再正規化して使う。 */
function evRoi(raceSamples: Scored[][], threshold: number, cal: Calibration | null): { roi: number; bets: number } {
  let bet = 0;
  let ret = 0;
  for (const race of raceSamples) {
    // 較正 → 再正規化（レース内合計=1を維持）。
    let probs = race.map((s) => s.rawProb);
    if (cal) {
      const c = race.map((s) => applyCalibration(s.rawProb, cal));
      const sum = c.reduce((a, b) => a + b, 0) || 1;
      probs = c.map((v) => v / sum);
    }
    race.forEach((s, i) => {
      const floor = 1 / s.fieldSize;
      if (probs[i]! < floor) return; // contenderOnly
      if (s.odds <= 0) return;
      const ev = probs[i]! * s.odds;
      if (ev >= threshold) {
        bet += 100;
        if (s.finish === 1) ret += s.odds * 100;
      }
    });
  }
  return { roi: bet > 0 ? (ret / bet) * 100 : 0, bets: bet / 100 };
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  if (args.input.length === 0) {
    console.error("使い方: npm run calibrate -- --input <CSV> [--input ...] --odds-col 49 [--split 0.7] [--min-horses 6]");
    process.exit(1);
    return;
  }
  const config = loadConfig(args.config);
  const races = buildDataset(args.input.map((p) => resolve(p)), { oddsCol: args.oddsCol, minHorses: args.minHorses });
  races.sort((a, b) => a.date.getTime() - b.date.getTime());
  const cut = Math.floor(races.length * args.splitRatio);
  const train = races.slice(0, cut);
  const test = races.slice(cut);
  const cutDate = test[0]?.date;
  console.log(`\n較正: train ${train.length}レース / test ${test.length}レース（分割日 ${cutDate?.toISOString().slice(0, 10)}）`);

  // train で較正を学習。
  const trainSamples = scoreAll(train, config).flat();
  const cal = fitIsotonic(trainSamples.map((s) => ({ p: s.rawProb, win: s.finish === 1 ? 1 : 0 })));

  // test でアウトオブサンプル評価（生 vs 較正）。
  const testRaces = scoreAll(test, config);
  const raw = evRoi(testRaces, config.ev.threshold, null);
  const calib = evRoi(testRaces, config.ev.threshold, cal);

  console.log("\n■ アウトオブサンプル（test期間）単勝EV≥1戦略の回収率");
  console.log(`  生 win_prob      : ${raw.roi.toFixed(1)}%  (${raw.bets}点)`);
  console.log(`  較正後 win_prob  : ${calib.roi.toFixed(1)}%  (${calib.bets}点)`);
  console.log(`  改善             : ${(calib.roi - raw.roi >= 0 ? "+" : "")}${(calib.roi - raw.roi).toFixed(1)}pt`);

  console.log("\n■ 学習した較正マップ（生確率→較正確率）");
  cal.x.forEach((x, i) => {
    if (i % 3 === 0 || i === cal.x.length - 1) console.log(`  ${(x * 100).toFixed(1)}% → ${(cal.y[i]! * 100).toFixed(1)}%`);
  });

  const outDir = resolve(args.outDir);
  mkdirSync(outDir, { recursive: true });
  writeFileSync(resolve(outDir, "calibration.json"), JSON.stringify(cal, null, 2), "utf-8");
  console.log(`\n較正マップ: ${outDir}/calibration.json\n`);
}

main();
