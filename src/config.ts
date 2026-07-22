import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

export interface RankThreshold {
  rank: string;
  min: number;
}

export interface Config {
  scoring: {
    base: { recencyWeights: number[]; scale: number };
    training: {
      evaluation: Record<string, number>;
      trend: Record<string, number>;
    };
    pedigree: { sireWeight: number; damSireWeight: number };
    softmaxTemperature: number;
    placeProb: { multiplier: number; cap: number };
  };
  ev: { threshold: number; evBands: number[]; contenderOnly: boolean };
  rank: { thresholds: RankThreshold[] };
  marks: {
    order: string[];
    crossCount: number;
    gapForCross: number;
  };
  rules: Record<string, number>;
  ruleParams: Record<string, number>;
  popularityColors: Record<string, string>;
  pastEvalColors: Record<string, string>;
}

/** config.json を読み込む。パスを渡さなければリポ直下の config.json。 */
export function loadConfig(path?: string): Config {
  const p = path ?? resolve(__dirname, "..", "config.json");
  const raw = readFileSync(p, "utf-8");
  return JSON.parse(raw) as Config;
}
