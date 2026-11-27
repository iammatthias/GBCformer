// verify-pixels.ts <frames.txt> <first frame of frames.txt> <pixel file FROM-TO.bgr0>...
// Each frame of a pixel file (the runner's --pixels: blue, green, red, zero per pixel) is
// hashed as the frame hash is (sha1 of the red, green and blue bytes) and held to the
// line of frames.txt for its frame, so a clip cut from the file shows the frames whose
// hashes were compared with the reference.
const [list, start, ...files] = process.argv.slice(2);
const hashes = (await Bun.file(list).text()).trim().split("\n");
const size = 160 * 144 * 4;
let bad = 0;
for (const path of files) {
  const from = Number(path.split("/").pop()!.split("-")[0]);
  const file = Bun.file(path);
  const frames = file.size / size;
  const rgb = new Uint8Array(160 * 144 * 3);
  let wrong = 0;
  for (let i = 0; i < frames; i++) {
    const px = new Uint8Array(await file.slice(i * size, (i + 1) * size).arrayBuffer());
    for (let p = 0; p < 160 * 144; p++) {
      rgb[p * 3] = px[p * 4 + 2];
      rgb[p * 3 + 1] = px[p * 4 + 1];
      rgb[p * 3 + 2] = px[p * 4];
    }
    if (new Bun.CryptoHasher("sha1").update(rgb).digest("hex") !== hashes[from + i - Number(start)]) wrong++;
  }
  console.log(`${wrong ? "FAIL " : "ok   "} ${path.split("/").slice(-3).join("/")}: ${frames} frames, ${wrong} with another hash than frames.txt`);
  bad += wrong;
}
process.exit(bad ? 1 : 0);
