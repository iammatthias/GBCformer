// bun digest.ts <out prefix> <name> < record: reduces one stream's pass record, read once from
// stdin (a FIFO the stream writes), to `<sha1> <lines> <name>` in <prefix>.sha1 and one
// 4-byte CRC-32 per line in <prefix>.lines. The sha1 is the comparison; `cmp` of two .lines
// files only locates the first differing pass (byte offset / 4 + 1) once the sha1s differ,
// so neither raw record is kept.
//
// DROP=<slot,slot,...> leaves those slots out of a stream whose header names them: their
// tokens in each event's sequence and their decoded positions. The positions left out are
// counted into <prefix>.dropped, so a comparison can require that one side decoded them and
// the other did not (the memory core's audit-only slots, `MemCheckpointCore::without_audit_slots`).
import { appendFileSync, writeFileSync } from "node:fs";

const [prefix, name] = process.argv.slice(2);
const dropNames = (process.env.DROP ?? "").split(",").filter((s) => s.length > 0);
const whole = new Bun.CryptoHasher("sha1");
const SIZE = 1 << 18;
const out = new Uint8Array(SIZE * 4);
const view = new DataView(out.buffer);
let held = 0;
let count = 0;
let carry: Uint8Array | null = null;
let header = true;
let drop: Set<number> | null = null;
let dropped = 0;
writeFileSync(`${prefix}.lines`, new Uint8Array(0));

function flush() {
  appendFileSync(`${prefix}.lines`, out.subarray(0, held * 4));
  held = 0;
}

function normalized(bytes: Uint8Array): Uint8Array {
  if (header) {
    header = false;
    if (dropNames.length > 0) {
      const h = JSON.parse(new TextDecoder().decode(bytes));
      const at: number[] = h.at ?? [];
      const picked = (h.slots as string[]).flatMap((s, j) => (dropNames.includes(s) && at[j] >= 0 ? [at[j]] : []));
      if (picked.length > 0) drop = new Set(picked);
    }
    return bytes;
  }
  if (drop === null) return bytes;
  const e = JSON.parse(new TextDecoder().decode(bytes));
  const keep = (e.positions as number[]).map((p) => !drop!.has(p));
  dropped += keep.filter((k) => !k).length;
  e.positions = (e.positions as number[]).filter((_, i) => keep[i]);
  e.out = (e.out as number[]).filter((_, i) => keep[i]);
  e.tokens = (e.tokens as number[]).filter((_, p) => !drop!.has(p));
  return new TextEncoder().encode(JSON.stringify(e));
}

function line(raw: Uint8Array) {
  const bytes = normalized(raw);
  whole.update(bytes);
  whole.update("\n");
  view.setUint32(held * 4, Bun.hash.crc32(bytes));
  count++;
  if (++held == SIZE) flush();
}

function join(a: Uint8Array, b: Uint8Array) {
  const j = new Uint8Array(a.length + b.length);
  j.set(a);
  j.set(b, a.length);
  return j;
}

for await (const chunk of Bun.stdin.stream()) {
  let start = 0;
  for (let nl = chunk.indexOf(10); nl >= 0; nl = chunk.indexOf(10, start)) {
    const piece = chunk.subarray(start, nl);
    line(carry ? join(carry, piece) : piece);
    carry = null;
    start = nl + 1;
  }
  if (start < chunk.length) carry = carry ? join(carry, chunk.subarray(start)) : chunk.slice(start);
}
if (carry) line(carry);
flush();
writeFileSync(`${prefix}.sha1`, `${whole.digest("hex")} ${count} ${name}\n`);
if (dropNames.length > 0) writeFileSync(`${prefix}.dropped`, `${dropped}\n`);
