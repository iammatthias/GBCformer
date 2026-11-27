// Sums the APU's and control's gate updates per family (GBCFORMER_UNITS=1 stderr), and splits the APU's into the mix and output stage and the channel timers.
// Usage: bun tools/census/families.ts <run stderr>
const t = await Bun.file(process.argv[2]).text();
for (const tag of ["apu gate updates per fast pass by node family", "control gate updates per fast pass by node family"]) {
  const line = t.split("\n").find((l) => l.startsWith(tag));
  if (!line) continue;
  const rows = line.split(": ")[1].split(", ").map((p) => { const i = p.lastIndexOf(" "); return [p.slice(0, i), Number(p.slice(i + 1))] as [string, number]; });
  const cats: [string, RegExp][] = [["mix and output (pan, volume, DAC sum, high-pass, sample bytes)", /^(AL|AR|GL|GR|cap|mix|FL|FR|SL|SR)/], ["channel timers and phases", /(T\d|t4|u3|ph2|T1|wave|sq|lfsr|smp)/], ["routing", /^route/]];
  const sums = new Map<string, number>();
  for (const [f, n] of rows) { const c = tag.startsWith("apu") ? cats.find(([, r]) => r.test(f))?.[0] ?? "other" : "listed"; sums.set(c, (sums.get(c) ?? 0) + n); }
  const total = rows.reduce((a, [, n]) => a + n, 0);
  console.log(`${tag.split(" ")[0]}: ${total.toFixed(0)} updates in the listed families; ${[...sums].map(([c, n]) => `${c} ${n.toFixed(0)}`).join("; ")}`);
}
