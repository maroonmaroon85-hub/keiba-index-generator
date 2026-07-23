import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { loadConfig } from "./config.js";
import { loadPreRaceFromJson, overrideRaceParams } from "./parser/json.js";
import { parseTargetCsv, type TargetRaceHeader } from "./parser/target-csv.js";
import { scoreRace } from "./scoring/score.js";
import { renderHtml } from "./render/html.js";
import { toScoreExport } from "./export.js";
import { buyList, winOddsFromHorses } from "./ev/ev.js";
import { exoticCombos, type ExoticType } from "./ev/harville.js";
import type { PreRaceData, Pace, TrackCondition, Surface } from "./model/pre-race.js";

interface Args {
  // JSONモード（Phase 1）
  input?: string;
  // TARGET CSVモード（Phase 2）
  shutuba?: string;
  seiseki?: string;
  raceId?: string;
  course?: string;
  surface?: Surface;
  distance?: number;
  date?: string;
  name?: string;
  // 共通（手入力）
  pace?: Pace;
  condition?: TrackCondition;
  outDir: string;
  config?: string;
}

function parseArgs(argv: string[]): Args {
  const args: Args = { outDir: "out" };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = argv[i + 1];
    switch (a) {
      case "--input": args.input = next; i++; break;
      case "--shutuba": args.shutuba = next; i++; break;
      case "--seiseki": args.seiseki = next; i++; break;
      case "--race-id": args.raceId = next; i++; break;
      case "--course": args.course = next; i++; break;
      case "--surface": args.surface = next as Surface; i++; break;
      case "--distance": args.distance = Number(next); i++; break;
      case "--date": args.date = next; i++; break;
      case "--name": args.name = next; i++; break;
      case "--pace": args.pace = next as Pace; i++; break;
      case "--condition": args.condition = next as TrackCondition; i++; break;
      case "--out": args.outDir = next ?? "out"; i++; break;
      case "--config": args.config = next; i++; break;
    }
  }
  return args;
}

const USAGE = `使い方:
  JSONダミー:  npm run generate -- --input <race.json> [--pace H] [--condition 良]
  TARGET実CSV: npm run generate -- --shutuba <出馬表.csv> --seiseki <成績.csv> \\
                 --course 函館 --surface ダ --distance 1700 --date 2026-07-19 \\
                 --pace M --condition 稍 [--race-id ...] [--name 駒場特別]
  共通: [--out out] [--config config.json]`;

function loadPreRace(args: Args): PreRaceData {
  if (args.shutuba || args.seiseki) {
    if (!args.shutuba || !args.seiseki) {
      throw new Error("TARGET CSVモードは --shutuba と --seiseki の両方が必要です");
    }
    if (!args.course || !args.surface || !args.distance || !args.date) {
      throw new Error("TARGET CSVモードは --course --surface --distance --date が必要です");
    }
    const header: TargetRaceHeader = {
      raceId: args.raceId ?? `${args.date.replace(/-/g, "")}_${args.course}`,
      course: args.course,
      surface: args.surface,
      distance: args.distance,
      pace: args.pace ?? "M",
      condition: args.condition ?? "良",
      date: args.date,
      ...(args.name ? { name: args.name } : {}),
    };
    return parseTargetCsv(resolve(args.shutuba), resolve(args.seiseki), header);
  }
  if (args.input) {
    const pre = loadPreRaceFromJson(resolve(args.input));
    return overrideRaceParams(pre, args.pace, args.condition);
  }
  throw new Error("入力がありません（--input か --shutuba/--seiseki）");
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  let pre: PreRaceData;
  try {
    pre = loadPreRace(args);
  } catch (e) {
    console.error((e as Error).message + "\n\n" + USAGE);
    process.exit(1);
    return;
  }

  const config = loadConfig(args.config);
  const scored = scoreRace(pre, config);
  const html = renderHtml(pre, scored, config);
  const json = toScoreExport(scored);

  const outDir = resolve(args.outDir);
  mkdirSync(outDir, { recursive: true });
  const htmlPath = resolve(outDir, `${pre.race.raceId}.html`);
  const jsonPath = resolve(outDir, `${pre.race.raceId}.score.json`);
  writeFileSync(htmlPath, html, "utf-8");
  writeFileSync(jsonPath, JSON.stringify(json, null, 2), "utf-8");

  // コンソールに要約と生成パスを出す。
  const ranked = [...scored.horses].sort((a, b) => b.score - a.score);
  console.log(`\n${pre.race.course}${pre.race.surface}${pre.race.distance}m  ペース${pre.race.pace} 馬場${pre.race.condition}  (${scored.horses.length}頭)`);
  console.log("─".repeat(52));
  for (const h of ranked) {
    const mark = h.mark || " ";
    console.log(
      `${mark} ${String(h.number).padStart(2)} ${h.name.padEnd(10)} ` +
        `評${h.rank.padEnd(2)} 指数${String(h.score).padStart(5)} ` +
        `単${(h.winProb * 100).toFixed(1)}% 複${(h.placeProb * 100).toFixed(1)}%`,
    );
  }
  console.log("─".repeat(52));

  // EV買い目（単勝。出馬表の単勝オッズを使用）。複勝は複勝オッズ受領後に対応。
  const bets = buyList(scored, winOddsFromHorses(pre.horses), config);
  if (bets.length > 0) {
    console.log(`\n■ 買い目（EV ≥ ${config.ev.threshold}）`);
    for (const b of bets) {
      console.log(`  ${b.type} ${String(b.number).padStart(2)} ${b.name.padEnd(10)} 確率${(b.prob * 100).toFixed(1)}% × オッズ${b.odds} = EV ${b.ev}`);
    }
  } else {
    console.log(`\n■ 買い目: なし（EV ≥ ${config.ev.threshold} の馬なし）`);
  }
  // 連系券種の確率上位候補（Harville）。オッズ無しでも「買うならこの組み合わせ」を提示。
  // EV判定は各券種オッズを渡せば可能（現状は確率順の候補まで）。
  console.log(`\n■ 連系の確率上位候補（オッズ照合前）`);
  const nameOf = new Map(pre.horses.map((h) => [h.number, h.name]));
  for (const t of ["馬連", "馬単", "ワイド", "三連複"] as ExoticType[]) {
    const combos = exoticCombos(scored, t, 3);
    const s = combos
      .map((c) => `${c.horses.join(t === "馬単" ? "→" : "-")}(${(c.prob * 100).toFixed(1)}%)`)
      .join("  ");
    console.log(`  ${t.padEnd(4)}: ${s}`);
  }
  void nameOf;
  console.log("─".repeat(52));
  console.log(`HTML: ${htmlPath}`);
  console.log(`JSON: ${jsonPath}\n`);
}

main();
