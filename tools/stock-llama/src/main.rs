//! GBCformer's one model in stock llama.cpp, through `shim.c`. `play` decodes the groups
//! the project's engine decodes and `diff` counts every token that differs.
//!
//! ```sh
//! stock-llama play <one-model.gguf|none|replay> <rom> --frames N [--hardware dmg|cgb|auto] [--warmup F] [--boot FILE] [--max-passes N] [--lanes N]
//! stock-llama diff <recording dir> <recording dir>
//! stock-llama state-diff <saved cache> <saved cache>
//! stock-llama scope <rom> [--hardware dmg|cgb|auto] [--warmup F]
//! stock-llama split-check
//! stock-llama shapes <model.gguf> --shapes NAME=TOKENS:ATTENDING:ROWS:CELLS,... [--cells N | --state FILE] [--reps N] [--lanes N]
//! ```

mod cache;
mod checkpoint;
mod cpp;
mod play;
mod runtime;
mod scope;
mod shapes;

unsafe extern "C" {
    fn _exit(code: i32) -> !;
}

/// Exits without the runtimes' static destructors (ggml's Metal device asserts that
/// every buffer was freed, and a shared runtime outlives the streams that use it).
pub fn quit(code: i32) -> ! {
    use std::io::Write;
    std::io::stdout().flush().ok();
    std::io::stderr().flush().ok();
    unsafe { _exit(code) }
}

/// The one checkpoint and each part's stream spec.
pub fn checkpoint_cfg() -> (gbcformer::llama::Model, Vec<gbcformer::engine::StreamSpec>) {
    let (m, specs, _, _) = checkpoint_parts();
    (m, specs)
}

/// `checkpoint_cfg`, the merged circuit it compiles and the GBC CPU's own circuit.
pub fn checkpoint_parts() -> (gbcformer::llama::Model, Vec<gbcformer::engine::StreamSpec>, gbcformer::circuit::Circuit, gbcformer::circuit::Circuit) {
    let parts = gbcformer::one_model::parts();
    let (c, specs) = gbcformer::one_model::circuit(&parts);
    let mut m = gbcformer::llama::Model::compile_mem(&c, gbcformer::memory::attn::Params::memory());
    gbcformer::gguf::from_env(&mut m);
    let cpu = parts.into_iter().nth(gbcformer::one_model::Part::Cpu as usize).expect("the GBC CPU's part");
    (m, specs, c, cpu)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str);
    let num = |name: &str, default: u64| flag(name).map_or(default, |v| v.parse().unwrap_or_else(|_| panic!("{name} {v}: not a number")));
    let threads = num("--threads", 8) as usize;
    let lanes = num("--lanes", 96) as usize;
    let hardware = match flag("--hardware") {
        None | Some("auto") => None,
        Some("dmg") => Some(gbcformer::Hardware::Dmg),
        Some("cgb") => Some(gbcformer::Hardware::Cgb),
        Some(other) => panic!("--hardware {other}: expected dmg, cgb or auto"),
    };
    match args.first().map(String::as_str) {
        Some("play") => play::run(play::Options {
            model: (args[1] != "none").then(|| args[1].clone()),
            rom: args[2].clone(),
            frames: num("--frames", 1),
            lockstep: flag("--lockstep").unwrap_or("all").to_string(),
            threads,
            max_passes: num("--max-passes", u64::MAX),
            warmup: num("--warmup", 0),
            lanes,
            hardware,
            boot: flag("--boot").map(str::to_string),
        }),
        Some("split-check") => play::split_check(),
        Some("shapes") => {
            let list: Vec<shapes::Shape> = flag("--shapes").expect("--shapes NAME=TOKENS:ATTENDING:ROWS:CELLS,...").split(',').map(shapes::Shape::parse).collect();
            shapes::run(&args[1], &list, num("--cells", 4096) as usize, flag("--state"), num("--reps", 30) as usize, lanes, threads)
        }
        Some("scope") => scope::report(&args[1], hardware, num("--warmup", 0)),
        Some("state-diff") => {
            if !checkpoint::state_diff(&args[1], &args[2]) {
                quit(1)
            }
        }
        Some("diff") => {
            if !play::diff(&args[1], &args[2]) {
                quit(1)
            }
        }
        _ => {
            eprintln!("usage: stock-llama play|diff|state-diff|scope|split-check|shapes ... (see src/main.rs)");
            std::process::exit(2);
        }
    }
    quit(0)
}
