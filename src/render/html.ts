import type { PreRaceData, PreRaceHorse } from "../model/pre-race.js";
import type { ScoredRace, ScoredHorse } from "../scoring/types.js";
import type { Config } from "../config.js";
import { lineColor } from "../pedigree/master.js";
import { styleLabel } from "../scoring/style.js";
import {
  pastEvalMark,
  rotationLabel,
  predictedPosition,
  jockeyLabel,
  trendArrow,
} from "./labels.js";

function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** 予想人気の背景色。1..5 は config、それ以外は無色。 */
function popularityColor(rank: number, config: Config): string {
  return config.popularityColors[String(rank)] ?? "transparent";
}

function pastEvalCell(horse: PreRaceHorse, config: Config): string {
  // 前走→三前の順で3走分。無い分は空。
  const cells: string[] = [];
  for (let i = 0; i < 3; i++) {
    const run = horse.pastRuns[i];
    if (!run) {
      cells.push(`<span class="pe pe-empty">-</span>`);
      continue;
    }
    const { mark, blinker } = pastEvalMark(run);
    const color = config.pastEvalColors[mark] ?? "#868e96";
    cells.push(
      `<span class="pe" style="color:${color}" title="${run.course}${run.surface}${run.distance} ${run.finish}着">${mark}${blinker ? "B" : ""}</span>`,
    );
  }
  return cells.join(" ");
}

function rankClass(rank: string): string {
  return "rk-" + rank.replace("+", "p");
}

/** ScoredRace + PreRaceData から静的HTML（外部依存なし・インラインCSS）を生成。 */
export function renderHtml(pre: PreRaceData, scored: ScoredRace, config: Config): string {
  const { race } = pre;
  const preByNumber = new Map<number, PreRaceHorse>();
  for (const h of pre.horses) preByNumber.set(h.number, h);

  // 表示は馬番昇順。
  const rows = [...scored.horses].sort((a, b) => a.number - b.number);

  const bodyRows = rows
    .map((s: ScoredHorse) => {
      const h = preByNumber.get(s.number)!;
      const popColor = popularityColor(s.predictedPopularity, config);
      const plus = s.flags.plus.length ? s.flags.plus.join(" ") : "";
      const minus = s.flags.minus.length ? s.flags.minus.join(" ") : "";
      return `
      <tr>
        <td class="c-horse">
          <span class="num">${h.number}</span>
          <span class="name">${esc(h.name)}${h.blinker ? '<span class="tag">[B]</span>' : ""}</span>
          <span class="jockey">${esc(jockeyLabel(h))}</span>
        </td>
        <td class="c-pop" style="background:${popColor}">${s.predictedPopularity}</td>
        <td class="c-mark">${s.mark}</td>
        <td class="c-rank ${rankClass(s.rank)}">${s.rank}<span class="score">${s.score}</span></td>
        <td class="c-rot">${esc(rotationLabel(h, s, race.distance))}</td>
        <td class="c-note">
          ${minus ? `<span class="minus">${esc(minus)}</span>` : ""}
          ${plus ? `<span class="plus">${esc(plus)}</span>` : ""}
        </td>
        <td class="c-past">${pastEvalCell(h, config)}</td>
        <td class="c-ped">
          <span class="line" style="background:${lineColor(h.sireLine)}">${esc(h.sireLine)}</span>
          <span class="line" style="background:${lineColor(h.damSireLine)}">${esc(h.damSireLine)}</span>
        </td>
        <td class="c-style">${s.style}<span class="sub">${styleLabel(s.style)}</span></td>
        <td class="c-pos">${esc(predictedPosition(s, h.frame, race.pace))}</td>
        <td class="c-train">${h.training.evaluation}<span class="arrow">${trendArrow(h.training.trend)}</span></td>
        <td class="c-farm">${esc(h.producer)}<span class="sub">${esc(h.trainer)}</span></td>
      </tr>`;
    })
    .join("");

  const title = `${race.course}${race.surface}${race.distance}m 指数表`;
  const subtitle = `${race.name ? esc(race.name) + " / " : ""}想定ペース ${race.pace} / 馬場 ${race.condition} / race_id ${race.raceId}`;

  return `<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)}</title>
<style>
  :root { color-scheme: light; }
  body { font-family: "Hiragino Kaku Gothic ProN", "Yu Gothic", Meiryo, sans-serif; margin: 16px; color: #212529; background: #fff; }
  h1 { font-size: 18px; margin: 0 0 2px; }
  .sub { color: #868e96; }
  .meta { font-size: 12px; color: #495057; margin-bottom: 10px; }
  table { border-collapse: collapse; width: 100%; font-size: 12px; }
  th, td { border: 1px solid #dee2e6; padding: 3px 5px; vertical-align: middle; }
  thead th { background: #343a40; color: #fff; font-weight: 600; position: sticky; top: 0; white-space: nowrap; }
  tbody tr:nth-child(even) { background: #f8f9fa; }
  .c-horse .num { display: inline-block; min-width: 18px; text-align: center; background: #495057; color: #fff; border-radius: 3px; padding: 0 3px; margin-right: 4px; }
  .c-horse .name { font-weight: 600; }
  .c-horse .jockey { display: block; color: #495057; font-size: 11px; }
  .c-horse .tag { color: #e8590c; font-size: 10px; margin-left: 2px; }
  .c-pop { text-align: center; font-weight: 700; }
  .c-mark { text-align: center; font-size: 16px; font-weight: 700; }
  .c-rank { text-align: center; font-weight: 700; }
  .c-rank .score { display: block; font-weight: 400; color: #868e96; font-size: 10px; }
  .rk-Sp { color: #d6336c; } .rk-S { color: #e8590c; } .rk-A { color: #1971c2; } .rk-B { color: #2f9e44; } .rk-C { color: #868e96; }
  .c-note { max-width: 150px; }
  .c-note .minus { display: block; color: #c92a2a; }
  .c-note .plus { display: block; color: #2b8a3e; }
  .c-past { text-align: center; white-space: nowrap; }
  .c-past .pe { font-weight: 700; margin: 0 1px; }
  .c-past .pe-empty { color: #ced4da; font-weight: 400; }
  .c-ped .line { display: block; border-radius: 2px; padding: 0 3px; margin: 1px 0; font-size: 11px; }
  .c-style { text-align: center; font-weight: 700; }
  .c-style .sub, .c-farm .sub, .c-train .arrow { font-weight: 400; }
  .c-style .sub, .c-farm .sub { display: block; color: #868e96; font-size: 10px; }
  .c-train { text-align: center; font-weight: 700; }
  .c-train .arrow { margin-left: 2px; }
  caption { caption-side: bottom; text-align: left; color: #adb5bd; font-size: 10px; padding-top: 6px; }
</style>
</head>
<body>
  <h1>${esc(title)}</h1>
  <div class="meta">${subtitle}</div>
  <table>
    <thead>
      <tr>
        <th>馬名/騎手</th>
        <th>人気</th>
        <th>印</th>
        <th>評価</th>
        <th>性齢/ローテ</th>
        <th>備考(−/＋)</th>
        <th>前走/二前/三前</th>
        <th>父系/母父系</th>
        <th>脚質</th>
        <th>予想位置</th>
        <th>調教</th>
        <th>生産地/厩舎</th>
      </tr>
    </thead>
    <tbody>${bodyRows}
    </tbody>
    <caption>keiba-index-generator (Phase 1 / ダミーデータ)。weight・閾値は仮置きで Phase 3 のバックテストで調整する。</caption>
  </table>
</body>
</html>
`;
}
