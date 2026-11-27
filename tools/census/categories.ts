// Groups GBC CPU circuit families into categories.
// Usage: bun tools/census/categories.ts static <levels --table cpu output> | dyn <run stderr with GBCFORMER_UNITS=1>
const cats: [string, RegExp][] = [
  ["opcode decode (nibble patterns, 256 one-hots, CB fields)", /^(OP0(=\*)?|OP0w|OP|cb\*|cbxy|cbw|cb\d?|len|MC|taken|jr|imm16h?|imm|irq\d?|ime|halt|disp\d?|cin|rd)$/],
  ["memory: byte select from a fetched line (OP1, OP2, D0, D1)", /^(OP[12]\[\d\]|D[01]\[\d\]|OP[12]w|D0r?w|D1w)$/],
  ["memory: write line merge (WL)", /^WL\d*$/],
  ["memory: address translation and physical selects", /^(F\d?|WA\d*|RHL|RSP|RP\d?|RRARE|WHL|WSP\d?|WRARE\d?|WA16?|WFFN|RFFN|RFFC|WP\d?|MAP[0RSW]?|WQ)$/],
  ["address muxes (RA, rare)", /^(RA\d?|rare\d?)$/],
  ["16-bit address adders (pc+n, sp±n, pairs, hl+pair, sp+e8)", /^(pc\+\d?|sp[+-]\d?|sp\+e\d?|pair[+-]?\d?|hl\+pair|a16\+\d?)$/],
  ["operand muxes (src, x, sel)", /^(src|x|sel)$/],
  ["ALU (add, sub, logic, inc/dec, daa, CB rot/bit/res)", /^(add|sub|and|or|or~|xor|inc|dec|daa[+-]?|cbres|cbz|bit|rot\d?|w)$/],
  ["output field muxes with keep (registers, flags, write fields)", /^([ABCDEHL]'|PC'|SP'|F'|CTL'|W\d?|WS|RQ)$/],
  ["routing", /^route$/],
  ["clearing", /^clear$/],
];
const mode = process.argv[2];
const text = await Bun.file(process.argv[3]).text();
let rows: [string, number][] = [];
if (mode === "static") {
  for (const line of text.split("\n").slice(2)) {
    const m = line.match(/^(\S+)\s+(\d+)\s/);
    if (m) rows.push([m[1], Number(m[2])]);
  }
} else {
  const line = text.split("\n").find((l) => l.includes("by node family"))!;
  for (const part of line.split(": ")[1].split(", ")) {
    const i = part.lastIndexOf(" ");
    rows.push([part.slice(0, i), Number(part.slice(i + 1))]);
  }
}
const sums = new Map<string, number>();
const unmatched: string[] = [];
let total = 0;
for (const [f, n] of rows) {
  total += n;
  const c = cats.find(([, r]) => r.test(f))?.[0];
  if (!c) unmatched.push(`${f} ${n}`);
  sums.set(c ?? "other", (sums.get(c ?? "other") ?? 0) + n);
}
for (const [c, n] of [...sums].sort((a, b) => b[1] - a[1])) console.log(`${n.toString().padStart(6)} ${(100 * n / total).toFixed(1).padStart(5)}%  ${c}`);
console.log(`total ${total}; unmatched: ${unmatched.join(", ")}`);
