// Projects llama.cpp ms per pass and hours to frame 300 for each redesign in
// docs/architecture-census.md §5, by scaling the H9 ablation components of
// docs/stock-llama.md §12a with each variant's dims. The results are estimates, not
// measurements.
// Usage: bun tools/census/project.ts
// `mult` is the passes per frame against today's (the serialised GBC CPU's M-cycle passes).
type Dims = { hidden: number; L: number; heads: number; units: number; lm: number; cells: number; mult?: number };
type Part = { name: string; passes: number; rows: number; bench: number; decode: number;
  comp: { attn: number; ffn: number; proj: number; lm: number; norm: number }; floor: number; base: Dims };
// `bench` is H11's ms per pass, and `decode` the part of it inside llama_decode (H11's split).
// `comp` is the H9 ablation, with the LM head cut to what H11 left.
const parts: Part[] = [
  { name: "CPU", passes: 981204, rows: 89, bench: 29.9, decode: 28.9, comp: { attn: 11.6, ffn: 8.0, proj: 6.6, lm: 0.6, norm: 1.8 }, floor: 2.6,
    base: { hidden: 5120, L: 15, heads: 157, units: 22463, lm: 4176, cells: 7300 } },
  { name: "PPU", passes: 83501, rows: 170, bench: 38.6, decode: 38.0, comp: { attn: 5.6, ffn: 19.7, proj: 8.2, lm: 0.3, norm: 2.2 }, floor: 2.0,
    base: { hidden: 5120, L: 15, heads: 157, units: 40615, lm: 256, cells: 2091 } },
  { name: "APU", passes: 243952, rows: 120, bench: 23.0, decode: 22.5, comp: { attn: 3.8, ffn: 7.8, proj: 6.6, lm: 0.2, norm: 1.2 }, floor: 1.5,
    base: { hidden: 5120, L: 15, heads: 157, units: 21142, lm: 256, cells: 2091 } },
  { name: "control", passes: 59052, rows: 82, bench: 21.6, decode: 18.0, comp: { attn: 3.4, ffn: 5.3, proj: 5.0, lm: 0.1, norm: 1.4 }, floor: 1.1,
    base: { hidden: 5120, L: 15, heads: 157, units: 14188, lm: 256, cells: 2091 } },
];
// The own units are `shape`'s live units for the GBC CPU and APU (measured), and estimates from their circuits for the PPU and control.
const own = { cpu: 13379, ppu: 31500, apu: 11473, ctl: 4200 };
const variants: [string, Partial<Dims>[]][] = [
  ["H11 one model (measured)", [{}, {}, {}, {}]],
  ["H12 head slices", [{ heads: 105 }, { heads: 148 }, { heads: 87 }, { heads: 82 }]],
  ["(a) overlay, hidden 2,624, 15 layers", [
    { hidden: 2624, heads: 18, units: own.cpu }, { hidden: 2624, heads: 129, units: own.ppu },
    { hidden: 2624, heads: 8, units: own.apu, cells: 0 }, { hidden: 2624, heads: 5, units: own.ctl, cells: 0 }]],
  ["(a') prefix overlay, early exit", [
    { hidden: 1792, L: 11, heads: 18, units: own.cpu }, { hidden: 2368, L: 15, heads: 129, units: own.ppu },
    { hidden: 1280, L: 14, heads: 8, units: own.apu, cells: 0 }, { hidden: 1216, L: 12, heads: 5, units: own.ctl, cells: 0 }]],
  ["(c) four checkpoints, as compiled", [
    { hidden: 2048, L: 11, heads: 18, units: own.cpu, lm: 3912 }, { hidden: 3072, L: 15, heads: 129, units: own.ppu },
    { hidden: 1024, L: 14, heads: 8, units: own.apu, cells: 0 }, { hidden: 1024, L: 12, heads: 5, units: own.ctl, cells: 0 }]],
  ["(c) four checkpoints, 64-rounded", [
    { hidden: 1664, L: 11, heads: 18, units: own.cpu, lm: 3912 }, { hidden: 2368, L: 15, heads: 129, units: own.ppu },
    { hidden: 1024, L: 14, heads: 8, units: own.apu, cells: 0 }, { hidden: 832, L: 12, heads: 5, units: own.ctl, cells: 0 }]],
  ["(a') + CPU d1 + d2", [
    { hidden: 1344, L: 9, heads: 20, units: 9200 }, { hidden: 2368, L: 15, heads: 129, units: own.ppu },
    { hidden: 1280, L: 14, heads: 8, units: own.apu, cells: 0 }, { hidden: 1216, L: 12, heads: 5, units: own.ctl, cells: 0 }]],
  // Estimated dims for docs/hardware-shaped.md §6, the hardware-shaped GBC CPU (prefetched opcode, bank keys
  // with one translated bus address, word-line selects) and PPU (each pixel row fetches
  // its own tile, palette and object pixels).
  ["(a') + hardware-shaped CPU", [
    { hidden: 1536, L: 11, heads: 17, units: 7500 }, { hidden: 2368, L: 15, heads: 129, units: own.ppu },
    { hidden: 1280, L: 14, heads: 8, units: own.apu, cells: 0 }, { hidden: 1216, L: 12, heads: 5, units: own.ctl, cells: 0 }]],
  ["(a') + hardware-shaped CPU and PPU", [
    { hidden: 1536, L: 11, heads: 17, units: 7500 }, { hidden: 1472, L: 15, heads: 60, units: 12500 },
    { hidden: 1280, L: 14, heads: 8, units: own.apu, cells: 0 }, { hidden: 1216, L: 12, heads: 5, units: own.ctl, cells: 0 }]],
  ["(a') + serialised CPU, one M-cycle a pass", [
    { hidden: 1024, L: 6, heads: 10, units: 4000, mult: 2.6 }, { hidden: 2368, L: 15, heads: 129, units: own.ppu },
    { hidden: 1280, L: 14, heads: 8, units: own.apu, cells: 0 }, { hidden: 1216, L: 12, heads: 5, units: own.ctl, cells: 0 }]],
];
const KERNEL_MS = 0.02; // A guess at the least time one Metal kernel costs, for the pessimistic bound.
for (const [name, over] of variants) {
  let hOpt = 0, hPes = 0; const cells: string[] = [];
  parts.forEach((p, i) => {
    const b = p.base, d = { ...b, ...over[i] };
    const ratio = {
      attn: (d.heads * (d.cells + p.rows)) / (b.heads * (b.cells + p.rows)),
      proj: (d.hidden * (d.heads + d.L)) / (b.hidden * (b.heads + b.L)),
      ffn: (d.hidden * d.units) / (b.hidden * b.units),
      lm: (d.hidden * d.lm) / (b.hidden * b.lm),
      norm: (d.hidden * d.L) / (b.hidden * b.L),
    };
    const kernels = { attn: d.L, proj: 4 * d.L, ffn: 2 * d.L, lm: 1, norm: 8 * d.L };
    const sum = Object.values(p.comp).reduce((a, x) => a + x, 0);
    const k = (p.decode - p.floor) / sum;
    let opt = 0, pes = 0;
    for (const c of Object.keys(p.comp) as (keyof typeof p.comp)[]) {
      opt += k * p.comp[c] * ratio[c];
      pes += Math.max(k * p.comp[c] * ratio[c], kernels[c] * KERNEL_MS);
    }
    const harness = p.bench - p.decode;
    const tOpt = p.floor + harness + opt, tPes = p.floor + harness + pes;
    const n = p.passes * (d.mult ?? 1);
    hOpt += (n * tOpt) / 3.6e6; hPes += (n * tPes) / 3.6e6;
    cells.push(`${p.name} ${tOpt.toFixed(1)}-${tPes.toFixed(1)}`);
  });
  console.log(`${name.padEnd(36)} ${cells.join(", ")} ms | ${hOpt.toFixed(1)}-${hPes.toFixed(1)} h`);
}
// Config::params: 2·vocab·d + L·(2·a·d + 2·kv·d + 3·d·i + 2·d) + d, head dim 64, one KV head.
const params = (d: number, L: number, heads: number, i: number, vocab: number) => 2 * vocab * d + L * (2 * heads * 64 * d + 2 * 64 * d + 3 * d * i + 2 * d) + d;
console.log(`dense: one model ${(params(5120, 15, 22, 6336, 36311) / 1e6).toFixed(1)} M; overlay 2,624 ${(params(2624, 15, 22, 6336, 36311) / 1e6).toFixed(1)} M; prefix overlay 2,368 shared slots ${(params(2368, 15, 22, 6336, 18422) / 1e6).toFixed(1)} M; CPU own 1,664 ${(params(1664, 11, 6, 1792, 8291) / 1e6).toFixed(1)} M; PPU own 2,368 ${(params(2368, 15, 20, 5504, 6090) / 1e6).toFixed(1)} M; control own 832 ${(params(832, 12, 5, 512, 10578) / 1e6).toFixed(1)} M; CPU after d1+d2 ${(params(1344, 9, 6, 1280, 8291) / 1e6).toFixed(1)} M`);
console.log(`dense, hardware-shaped: one model (prefix overlay 1,536, 15 layers, 12 heads, intermediate 3,520, shared slots) ${(params(1536, 15, 12, 3520, 18422) / 1e6).toFixed(1)} M; CPU own 1,472 ${(params(1472, 11, 6, 1024, 8291) / 1e6).toFixed(1)} M; PPU own 1,472 ${(params(1472, 15, 6, 2048, 6090) / 1e6).toFixed(1)} M`);
