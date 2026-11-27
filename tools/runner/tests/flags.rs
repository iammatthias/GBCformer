//! The opt-in flags `--presses FILE` and `--pixels DIR --clip FROM:TO`, as far as the
//! options go: the binary reads them before it opens the model, so these run without one.

use std::process::Command;

/// The runner's error for `args` after a model path that is never opened.
fn refusal(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_gbcformer-runner")).arg("no-model.gguf").args(args).output().expect("the runner starts");
    assert_eq!(out.status.code(), Some(1), "{args:?} is refused");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn press_file(name: &str, lines: &str) -> String {
    let path = std::env::temp_dir().join(format!("gbcformer-runner-flags-{}-{name}", std::process::id()));
    std::fs::write(&path, lines).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn each_line_of_a_press_file_is_a_press() {
    let good = press_file("good", "10:a:4\n20:down\n");
    assert!(refusal(&["--presses", &good, "--frames", "1"]).starts_with("usage:"), "two good lines pass, and the missing --rom is what stops the run");
    let bad = press_file("bad", "10:a:4\n20:down:0\n");
    assert_eq!(refusal(&["--presses", &bad, "--rom", "r"]), "20:down:0: HOLD must be at least 1 frame\n");
    let worse = press_file("worse", "10:a:4\nleft\n");
    assert_eq!(refusal(&["--rom", "r", "--presses", &worse]), "left: expected FRAME:BUTTON[:HOLD]\n");
    for f in [good, bad, worse] {
        std::fs::remove_file(f).unwrap();
    }
}

#[test]
fn a_press_file_that_cannot_be_read_is_named() {
    let e = refusal(&["--presses", "no-such-press-file", "--rom", "r"]);
    assert!(e.starts_with("no-such-press-file: "), "{e}");
    assert!(refusal(&["--rom", "r", "--presses"]).starts_with("usage:"));
}

#[test]
fn a_clip_is_a_range_of_frames() {
    assert_eq!(refusal(&["--rom", "r", "--pixels", "d", "--clip", "5"]), "5: expected FROM:TO\n");
    assert_eq!(refusal(&["--rom", "r", "--pixels", "d", "--clip", "5:x"]), "5:x: expected FROM:TO\n");
}
