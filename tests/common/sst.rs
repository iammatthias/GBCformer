use std::path::Path;

/// The SingleStepTests cases, or `None` with a line saying where they come from. The
/// repository holds none of them, and `tools/gate.sh` fails without them.
pub fn dir() -> Option<&'static Path> {
    let dir = Path::new("vendor/sm83/v1");
    if dir.is_dir() {
        return Some(dir);
    }
    eprintln!("skip: no vendor/sm83/v1. Clone https://github.com/SingleStepTests/sm83 (commit f9c30210) as vendor/sm83 to run this test");
    None
}
