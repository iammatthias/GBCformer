// Sums per-pass wire changes (the GBCFORMER_WIRE_CHANGES table, "name rate" lines) into GBC CPU categories.
// Usage: bun tools/census/changes.ts <table>
const cats: [string, RegExp][] = [
  ["inputs (registers, PC, SP, REGS, IE/IF)", /^(A|B|C|D|E|F|H|L|PCL|PCH|SPL|SPH|CTL|REGS\d*|IE|IF|D0|D1|I\d+)$/],
  ["memory heads' line bytes (fetch, data, write-line reads)", /^(OP\dh|D\dh|D\dr|WL\dh|.*h)$/],
  ["opcode decode (nibble patterns, 256 one-hots, CB fields)", /^(OP0|cb\d?|cbxy|cbw|len|MC|taken|jr|imm16h?|imm|irq\d?|ime|halt|disp\d?|cin|rd)$/],
  ["memory: byte select from a fetched line", /^(OP[12]\[\d\]|D[01]\[\d\])$/],
  ["memory: write line merge (WL)", /^WL\d*$/],
  ["memory: address translation and physical selects", /^(F\d?F?|RHL|RSP|RP\d?|RRARE|WHL|WSP\d?|WRARE\d?|WA16?|WFFN|RFFN|RFFC|WP\d?|MAP[0RSW]?|WQ)$/],
  ["address muxes (RA, rare)", /^(RA\d?|rare\d?)$/],
  ["16-bit address adders", /^(pc\+\d?|sp[+-]\d?|sp\+e\d?|pair[+-]?\d?|hl\+pair|a16\+\d?)$/],
  ["operand muxes (src, x, sel)", /^(src|x|sel(#\d+)?)$/],
  ["ALU", /^(add|sub|and|or|or~|xor|inc|dec|daa[+-]?|cbres|cbz|bit|rot\d?|w)$/],
  ["output field muxes with keep", /^([ABCDEHL]'|PC'|SP'|F'|CTL'|W\d?|WS|RQ)$/],
];
const rows = (await Bun.file(process.argv[2]).text()).trim().split("\n").map((l) => { const i = l.lastIndexOf(" "); return [l.slice(0, i), Number(l.slice(i + 1))] as [string, number]; });
const sums = new Map<string, number>();
const other = new Map<string, number>();
let total = 0;
for (const [name, r] of rows) {
  total += r;
  const stem = name.split(/[.]/)[0];
  const c = cats.find(([, re]) => re.test(stem))?.[0];
  if (!c) other.set(stem, (other.get(stem) ?? 0) + r);
  sums.set(c ?? "other", (sums.get(c ?? "other") ?? 0) + r);
}
for (const [c, n] of [...sums].sort((a, b) => b[1] - a[1])) console.log(`${n.toFixed(1).padStart(7)} ${(100 * n / total).toFixed(1).padStart(5)}%  ${c}`);
console.log(`total ${total.toFixed(1)} changes per pass; other: ${[...other].sort((a, b) => b[1] - a[1]).slice(0, 25).map(([k, v]) => `${k} ${v.toFixed(1)}`).join(", ")}`);
