//! The minimal inference runner (`docs/minimal-runner.md`). It loads the one model from
//! its GGUF alone, starts every stream from a snapshot given as data, and moves bytes
//! between the streams as the GGUF's wiring table says. It links the inference core and
//! nothing of the machine the model computes: no device, no bus map, no rule.
//!
//! `gbcformer-runner <model.gguf> --snapshot DIR --frames N [--press FRAME:BUTTON[:HOLD]]...
//! [--time START[:FRAMES_A_SECOND]] [--outputs DIR] [--presses FILE] [--pixels DIR --clip FROM:TO...]`
//!
//! `--outputs DIR` writes every frame's hash (`frames.txt`), every sample played
//! (`audio.raw`) and every save (`save-<n>.bin`), the files the full program writes with
//! `GBCFORMER_OUTPUTS`. Built with `--features record`, `GBCFORMER_RECORD=<dir>` writes
//! the engine's pass record, as the full program's does. `--presses FILE` reads a
//! `--press` from each line of FILE, and `--pixels DIR` with `--clip FROM:TO` writes the
//! frames shown from FROM up to TO (`clips.rs`).

mod clips;
mod length;
mod lines;
mod ports;
mod power_on;
mod present;
mod rows;
mod scheduler;
mod segments;
mod slots;
mod spans;
mod steps;
mod stepper;
mod worker;

use gbcformer::engine::{Engine, Video, VideoRecords};
use gbcformer::memory::image::Image;
use gbcformer::transport::{frame_sha1, Outputs};
use gbcformer::wiring::{Manifest, MemoryFile, Snapshot};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Options {
    model: PathBuf,
    snapshot: Option<PathBuf>,
    /// Compare the power-on state with the full program's snapshot in this directory.
    check: Option<PathBuf>,
    frames: u64,
    presses: Vec<(u64, String, u64)>,
    time: Option<(u64, u64)>,
    outputs: Option<PathBuf>,
    rom: PathBuf,
    boot: Option<PathBuf>,
    /// The segments and lines streams, and the memory cells' rows, each on a thread of its own.
    threads: bool,
    clips: clips::Clips,
}

fn parse() -> Result<Options, String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: gbcformer-runner <model.gguf> --rom FILE [--boot FILE] [--snapshot DIR | --check DIR] --frames N [--press FRAME:BUTTON[:HOLD]]... [--presses FILE] [--time START[:FRAMES_A_SECOND]] [--outputs DIR] [--pixels DIR --clip FROM:TO...] [--threads 0|1]";
    // Each line of a press file is a press.
    while let Some(i) = args.iter().position(|a| a == "--presses") {
        let file = args.get(i + 1).ok_or(usage)?;
        let lines = std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;
        args.splice(i..(i + 2).min(args.len()), lines.lines().flat_map(|l| ["--press".to_string(), l.to_string()]));
    }
    let mut clips = clips::Clips { dir: PathBuf::new(), ranges: Vec::new() };
    let mut it = args.iter();
    let model = PathBuf::from(it.next().ok_or(usage)?);
    let (mut snapshot, mut frames, mut presses, mut time, mut outputs, mut rom, mut boot, mut threads, mut check) = (None, None, Vec::new(), None, None, None, None, true, None);
    while let Some(a) = it.next() {
        let mut value = || it.next().cloned().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--snapshot" => snapshot = Some(PathBuf::from(value()?)),
            "--frames" => frames = Some(value()?.parse().map_err(|_| "--frames N")?),
            "--press" => {
                let v = value()?;
                let parts: Vec<&str> = v.split(':').collect();
                let bad = || format!("{v}: expected FRAME:BUTTON[:HOLD]");
                let (frame, name, hold) = match parts[..] {
                    [f, n] => (f, n, "3"),
                    [f, n, h] => (f, n, h),
                    _ => return Err(bad()),
                };
                let hold: u64 = hold.parse().map_err(|_| bad())?;
                if hold == 0 {
                    return Err(format!("{v}: HOLD must be at least 1 frame"));
                }
                presses.push((frame.parse().map_err(|_| bad())?, name.to_lowercase(), hold));
            }
            "--time" => {
                let v = value()?;
                let (s, per) = v.split_once(':').unwrap_or((&v, "60"));
                time = Some((s.parse().map_err(|_| "--time START[:FRAMES_A_SECOND]")?, per.parse().map_err(|_| "--time START[:FRAMES_A_SECOND]")?));
            }
            "--outputs" => outputs = Some(PathBuf::from(value()?)),
            "--rom" => rom = Some(PathBuf::from(value()?)),
            "--boot" => boot = Some(PathBuf::from(value()?)),
            "--threads" => threads = value()? != "0",
            "--check" => check = Some(PathBuf::from(value()?)),
            "--pixels" => clips.dir = PathBuf::from(value()?),
            "--clip" => {
                let v = value()?;
                let range = v.split_once(':').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)));
                clips.ranges.push(range.ok_or(format!("{v}: expected FROM:TO"))?);
            }
            other => return Err(format!("{other}: unknown flag\n{usage}")),
        }
    }
    Ok(Options { model, snapshot, check, frames: frames.unwrap_or(0), presses, time, outputs, rom: rom.ok_or(usage)?, boot, threads, clips })
}

fn image(dir: &Path, f: &MemoryFile) -> Result<Image, String> {
    let read = |name: &str| std::fs::read(dir.join(name)).map_err(|e| format!("{}: {e}", dir.join(name).display()));
    Image::from_parts(read(&f.bytes)?, read(&f.present)?.into_iter().map(|b| b != 0).collect(), f.lengths)
}

/// Fills `cells` (base, length) of `image` from the file at `path`, which must be that long.
fn fill(image: &mut Image, cells: (u32, usize), path: &Path) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < cells.1 {
        return Err(format!("{}: {} bytes, the snapshot holds {}", path.display(), bytes.len(), cells.1));
    }
    image.fill(cells.0, &bytes[..cells.1]);
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    gbcformer::qos::interactive();
    let o = parse()?;
    let t = Instant::now();
    let (manifest, wiring) = Manifest::read(&o.model)?;
    let (snap, memory, own, len, frames) = match &o.snapshot {
        Some(dir) => {
            let s = from_snapshot(dir, &o, &wiring.power_on)?;
            manifest.check_snapshot(&s.0)?;
            s
        }
        None => {
            let path = o.boot.as_ref().ok_or("without --snapshot the run starts at power-on: it needs --boot")?;
            let read = |p: &Path| std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()));
            power_on::build(&wiring.power_on, &read(&o.rom)?, &read(path)?)?
        }
    };
    if let Some(dir) = &o.check {
        return check(dir, &o, &wiring.power_on, (&snap, &memory, &own, &len, &frames));
    }
    let (model, specs, loaded) = gbcformer::gguf::open_model(&o.model)?;
    // The streams in the order the full program builds them, so the records' names match.
    let order = [wiring.steps.stream, wiring.lines.stream, wiring.segments.stream, wiring.spans.stream, wiring.length.stream, wiring.stepper.stream];
    let chosen: Vec<_> = order.iter().map(|&i| specs[i].clone()).collect();
    let mut engines = Engine::streams_with(model, &chosen, &|_, _| {}).into_iter();
    let (mut memory_stream, mut line_engine, segment_engine, span_engine, mut length_engine, mut stepper_engine) = (engines.next().unwrap(), engines.next().unwrap(), engines.next().unwrap(), engines.next().unwrap(), engines.next().unwrap(), engines.next().unwrap());
    let records = VideoRecords::default();
    memory_stream.insert(memory);
    let cache = memory_stream.memory.as_mut().expect("the steps stream holds the memory");
    cache.video = Some(Video { records: records.clone(), cells: vec![wiring.lines.mode_cell], ranges: wiring.lines.follows.clone() });
    // The cells' rows on a thread of their own: the same rows reach the same cells, later.
    if o.threads {
        cache.cells.defer();
    }
    // The stepper's memory is the lines stream's own.
    stepper_engine.insert(own.clone());
    line_engine.insert(own);
    length_engine.insert(len);
    let length = length::LengthStream::new(slots::Seq::new(length_engine), &wiring.length, &wiring.lines, &snap);
    let screen = present::Screen::new(&wiring.present, snap.colours, &frames);
    let segments = segments::Queued::new(worker::Worker::new(segments::SegmentStream::new(slots::Seq::new(segment_engine), &wiring.segments, &snap), o.threads));
    let spans = spans::SpanStream::new(slots::Seq::new(span_engine), &wiring.spans, &snap);
    let steps = steps::StepStream::new(slots::Seq::new(memory_stream), &wiring.steps, &snap, &|slot| spans.position(slot), &|slot| spans.output_position(slot));
    let lines = worker::Worker::new(lines::LineStream::new(slots::Seq::new(line_engine), &wiring.lines, &snap, screen, wiring.steps.line_bits), o.threads);
    let rows = rows::Rows::new(stepper::StepperStream::new(slots::Seq::new(stepper_engine), &wiring.stepper, &wiring.lines, &snap), snap.line_held, wiring.lines.own_memory);
    let mut run = scheduler::Run::new(steps, spans, segments, lines, length, rows, &wiring, &snap, records);
    eprintln!("runner: {} tensors from {} alone, six streams from the snapshot at frame {} ({:.1}s)", loaded.tensors, o.model.display(), snap.frame, t.elapsed().as_secs_f64());
    let button = |name: &str| wiring.present.buttons.iter().find(|(n, _)| n == name).map(|&(_, b)| b).ok_or(format!("unknown button {name}"));
    let presses: Vec<(u64, u8, u64)> = o.presses.iter().map(|(f, n, h)| button(n).map(|b| (*f, b, *h))).collect::<Result<_, _>>()?;
    let mut out = o.outputs.as_ref().map(|d| Outputs::new(d)).transpose()?;
    let mut filter = gbcformer::transport::HighPass { charge: snap.filter, gain: wiring.present.gain, shift: wiring.present.shift };
    let mut audio = Vec::new();
    let mut saves = 0;
    let mut played = 0;
    let start = Instant::now();
    for frame in snap.frame..o.frames {
        let now = o.time.map_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()), |(s, per)| s + frame / per.max(1));
        run.set_time(now);
        let pressed = presses.iter().filter(|&&(f, _, h)| frame >= f && frame - f < h).fold(0, |b, p| b | p.1);
        if let Some(save) = run.run_frame(pressed) {
            saves += 1;
            if let Some(out) = &mut out {
                out.save(&save)?;
            }
        }
        audio.clear();
        run.drain_audio(&mut filter, &mut audio, false);
        if let Some(out) = &mut out {
            out.audio(&audio)?;
            out.frame(&run.lines.call(|l| l.screen.shown().to_vec()))?;
        }
        if o.clips.wants(frame) {
            o.clips.frame(frame, played, &run.lines.call(|l| l.screen.shown().to_vec()))?;
        }
        played += audio.len() as u64;
    }
    let last = run.lines.call(|l| l.screen.shown().to_vec());
    audio.clear();
    run.drain_audio(&mut filter, &mut audio, true);
    // A check of the cells' rows that failed on their thread stops the run here, before it reports.
    run.steps.settle();
    let elapsed = start.elapsed().as_secs_f64();
    if let Some(out) = &mut out {
        out.audio(&audio)?;
    }
    if let Some(save) = run.take_save(false) {
        saves += 1;
        if let Some(out) = &mut out {
            out.save(&save)?;
        }
    }
    let ran = o.frames - snap.frame;
    let c = run.counts;
    println!("frames={ran} saves={saves} steps={} step_cycles={} idle_cycles={} stall_cycles={} pause_cycles={} rewrites={} rest_passes={} lock_pairs_differ={} compat_closes={} forwarded={} revised_before={} taken_after={} frame_sha1={}", c.steps, c.step_cycles, c.idle_cycles, c.stall_cycles, c.pause_cycles, run.steps.rewrite_passes, run.spans.rest_passes, run.spans.pairs_differ, run.rows.compat_closes, run.rows.forwarded, run.rows.revised_before, run.rows.taken_after, frame_sha1(&last));
    eprintln!("{:.1} fps over {elapsed:.2}s", ran as f64 / elapsed);
    let serial = run.spans.serial.clone();
    let scheduler::Run { steps, spans, segments, lines, length, .. } = run;
    let (segments, lines) = (segments.finish(), lines.finish());
    for (name, seq) in [("steps", &steps.seq), ("lines", &lines.seq), ("segments", &segments.p.seq), ("spans", &spans.seq), ("length", &length.seq)] {
        let s = &seq.engine.stats;
        eprintln!("{name}: {} passes, {} fast, {} dense fallbacks; cell reads {} by lookup, {} by full scan, {} lookups checked by the scan", s.passes, s.fast_passes, s.fallback_passes, s.cell_lookups, s.cell_scans, s.cell_oracle_scans);
    }
    if !serial.is_empty() {
        println!("serial: {}", String::from_utf8_lossy(&serial).trim_end());
    }
    Ok(())
}

/// The state a snapshot directory holds, the ROM and boot ROM filled in.
fn from_snapshot(dir: &Path, o: &Options, w: &gbcformer::wiring::PowerOn) -> Result<(Snapshot, Image, Image, Image, Vec<u32>), String> {
    let snap: Snapshot = serde_json::from_slice(&std::fs::read(dir.join("snapshot.json")).map_err(|e| format!("{}: {e}", dir.display()))?).map_err(|e| format!("snapshot.json: {e}"))?;
    let mut memory = image(dir, &snap.memory)?;
    fill(&mut memory, snap.rom, &o.rom)?;
    match (snap.boot, &o.boot) {
        (Some(cells), Some(path)) => {
            let read = |p: &Path| std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()));
            memory.fill(cells.0, &power_on::boot_image(w, &read(&o.rom)?, &read(path)?, cells.1)?);
        }
        (Some(_), None) => return Err("the snapshot is at power-on: it needs --boot".into()),
        (None, Some(_)) => return Err("the snapshot is past the boot ROM: --boot is not used".into()),
        (None, None) => {}
    }
    let mut own = image(dir, &snap.own)?;
    fill(&mut own, snap.own_rom, &o.rom)?;
    let mut len = image(dir, &snap.len)?;
    fill(&mut len, snap.own_rom, &o.rom)?;
    let frames = std::fs::read(dir.join(&snap.frames)).map_err(|e| e.to_string())?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    Ok((snap, memory, own, len, frames))
}

/// `--check DIR`: the state built from power-on against the full program's power-on
/// snapshot in DIR, every field and every byte.
fn check(dir: &Path, o: &Options, w: &gbcformer::wiring::PowerOn, (snap, memory, own, len, frames): (&Snapshot, &Image, &Image, &Image, &Vec<u32>)) -> Result<(), String> {
    let (want, wmem, wown, wlen, wframes) = from_snapshot(dir, o, w)?;
    let mut diffs = Vec::new();
    // The file names and the image lengths are how each was stored, not state.
    let strip = |s: &Snapshot| Snapshot { memory: want.memory.clone(), own: want.own.clone(), len: want.len.clone(), frames: want.frames.clone(), ..s.clone() };
    if strip(snap) != want {
        diffs.push(format!("snapshot fields: built {:?}\n want {:?}", strip(snap), want));
    }
    for (name, a, b) in [("memory", memory, &wmem), ("own memory", own, &wown), ("length memory", len, &wlen)] {
        let ((ab, ap, al), (bb, bp, bl)) = (a.parts(), b.parts());
        if let Some(p) = (0..ab.len().max(bb.len())).find(|&p| ab.get(p) != bb.get(p)) {
            diffs.push(format!("{name}: byte {p:#x} built {:?}, want {:?}", ab.get(p), bb.get(p)));
        }
        if let Some(p) = (0..ap.len().max(bp.len())).find(|&p| ap.get(p) != bp.get(p)) {
            diffs.push(format!("{name}: token flag {p:#x} built {:?}, want {:?}", ap.get(p), bp.get(p)));
        }
        if al != bl {
            diffs.push(format!("{name}: lengths built {al:?}, want {bl:?}"));
        }
    }
    if *frames != wframes {
        diffs.push("frames differ".into());
    }
    if diffs.is_empty() {
        println!("power-on state equal to {}: every snapshot field, {} memory bytes and token flags, {} own-memory bytes, both frames", dir.display(), memory.parts().0.len(), own.parts().0.len());
        Ok(())
    } else {
        Err(diffs.join("\n"))
    }
}
