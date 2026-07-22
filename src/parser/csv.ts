import { readFileSync } from "node:fs";
import iconv from "iconv-lite";

/** Shift_JIS の CSV を読み、行×列に分解（ダブルクォート対応、空行除去）。 */
export function readCsvShiftJis(path: string): string[][] {
  let buf: Buffer;
  try {
    buf = readFileSync(path);
  } catch {
    throw new Error(`CSVを読めません: ${path}`);
  }
  const text = iconv.decode(buf, "Shift_JIS");
  return text
    .split(/\r?\n/)
    .filter((line) => line.trim() !== "")
    .map(splitCsvLine);
}

/** 1行をカンマ分割（"..." 内のカンマは無視）。値の前後空白と囲みクォートは除去。 */
export function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQuote = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      inQuote = !inQuote;
    } else if (ch === "," && !inQuote) {
      out.push(cur.trim());
      cur = "";
    } else {
      cur += ch;
    }
  }
  out.push(cur.trim());
  return out;
}

export function toInt(v: string | undefined): number {
  const n = parseInt((v ?? "").trim(), 10);
  return Number.isFinite(n) ? n : 0;
}
export function toNum(v: string | undefined): number {
  const n = parseFloat((v ?? "").trim());
  return Number.isFinite(n) ? n : 0;
}
