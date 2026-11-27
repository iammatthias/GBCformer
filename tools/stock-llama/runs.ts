// usage: bun tools/stock-llama/runs.ts <model.gguf>
// Each stream's runs per layer from the GGUF's gbcformer.stream.* keys, and the other gbcformer.* keys: the FFN
// units (first unit + units) and the heads (first head + heads).
// stream-slices.patch's mm_cols starts a run's view of the weights on the 4 weights at or before its first
// one, so a run that starts `lead` weights past a multiple of 4 is read as a view lead + units wide (the padded
// view). Metal's matmul kernels load a weight row in blocks of 16 when the view is a multiple of 32 wide, and
// in fours when it is 32 to 63 wide and a multiple of 4; any other width takes element loads. A run is marked:
//   "block load": the view is a multiple of 32 wide;  "half4": the view is 32 to 63 wide and a multiple of 4;
//   "PADDED" before either: the view has a lead, so the kernel reads zero-padded activations too.
// An unaligned run of a multiple of 32 units is the form Metal misread before mm_cols aligned it
// (docs/perf-log.md, "The slices' NaN on Metal"); its padded view is no longer a multiple of 32 wide.
import { openSync, readSync } from "fs";
const fd = openSync(process.argv[2], "r");
const buf = Buffer.alloc(64 << 20);
readSync(fd, buf, 0, buf.length, 0);
let at = 0;
const u32 = () => { const v = buf.readUInt32LE(at); at += 4; return v; };
const u64 = () => { const v = Number(buf.readBigUInt64LE(at)); at += 8; return v; };
const str = () => { const n = u64(); const s = buf.toString("utf8", at, at + n); at += n; return s; };
const size: Record<number, number> = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
const val = (t: number): any => {
  if (t === 8) return str();
  if (t === 9) { const et = u32(); const n = u64(); const out: any[] = []; if (et === 4) { for (let i = 0; i < n; i++) out.push(u32()); return out; } for (let i = 0; i < n; i++) { const v = val(et); if (i < 4) out.push(v); } return out; }
  if (t === 4) return u32();
  const v = buf.readUIntLE(at, Math.min(size[t], 6)); at += size[t]; return v;
};
u32(); u32(); const nt = u64(); const nkv = u64();
const kv: Record<string, any> = {};
for (let i = 0; i < nkv; i++) { const k = str(); const t = u32(); kv[k] = val(t); }
for (const k of Object.keys(kv)) if (k.startsWith("gbcformer.") && !k.endsWith(".ffn") && !k.endsWith(".heads")) console.log(k, JSON.stringify(kv[k]).slice(0, 200));
console.log("feed_forward_length", JSON.stringify(kv["llama.feed_forward_length"]));
console.log("head_count", JSON.stringify(kv["llama.attention.head_count"]), "key_length", kv["llama.attention.key_length"]);
const headDim = kv["llama.attention.key_length"];
const flagged: string[] = [];
const headViews: string[] = [];
// One run of columns as mm_cols reads it: `first` and `cols` in weights.
const mark = (where: string, first: number, cols: number, list = true) => {
  const lead = first % 4, view = lead + cols;
  const kernel = view % 32 === 0 ? "block load" : view >= 32 && view <= 63 && view % 4 === 0 ? "half4" : "";
  if (kernel && list) flagged.push(`${where}: ${first}+${cols}, view ${view} wide, ${lead ? "PADDED " : ""}${kernel}`);
  return (lead ? ` (lead ${lead}, view ${view})` : "") + (kernel ? ` <<< ${lead ? "PADDED " : ""}${kernel}` : "");
};
for (const name of ["ppu", "cpu", "apu", "ctl", "len", "stp"]) {
  const f = kv[`gbcformer.stream.${name}.ffn`]; if (!f) continue;
  const h = kv[`gbcformer.stream.${name}.heads`];
  let p = 0, q = 0; const lines: string[] = [];
  for (let l = 0; l < kv[`gbcformer.stream.${name}.block_count`]; l++) {
    const runs = f[p++]; const rs: string[] = [];
    for (let r = 0; r < runs; r++, p += 2) rs.push(`${f[p]}+${f[p + 1]}${mark(`${name} L${l} ffn`, f[p], f[p + 1])}`);
    const hs: string[] = [];
    // The heads key lists each layer's head indices in rising order; consecutive ones are one run (the patch's loader).
    if (h) { const n = h[q++]; const runs: number[][] = []; for (let i = 0; i < n; i++, q++) { const last = runs[runs.length - 1]; if (last && last[0] + last[1] === h[q]) last[1]++; else runs.push([h[q], 1]); } for (const [first, heads] of runs) { hs.push(`${first}+${heads}`); if (headDim) headViews.push(mark(`${name} L${l} heads`, first * headDim, heads * headDim, false)); } }
    lines.push(`  L${l}: ffn ${rs.join(", ") || "none"}; heads ${hs.join(", ") || "none"}`);
  }
  console.log(name + "\n" + lines.join("\n"));
}
console.log(`\nhead runs: ${headViews.length}, each a whole number of heads of ${headDim} weights from a multiple of ${headDim}: ${headViews.filter((s) => s.includes("lead")).length} with a lead (padded), ${headViews.filter((s) => s.includes("block load")).length} block loads`);
console.log(`FFN views Metal's block-load or half4 kernels read (${flagged.length}, ${flagged.filter((s) => s.includes("PADDED")).length} of them padded):`);
for (const s of flagged) console.log("  " + s);
