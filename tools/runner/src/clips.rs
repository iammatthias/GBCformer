//! `--pixels DIR --clip FROM:TO`: the frames shown in each range, as files. Each range
//! is one file of raw pixels, four bytes a pixel (blue, green, red, zero), frame after
//! frame, and a text file with the count of samples played before its first frame.

use std::io::Write;
use std::path::PathBuf;

pub struct Clips {
    pub dir: PathBuf,
    /// First frame and the frame after the last.
    pub ranges: Vec<(u64, u64)>,
}

impl Clips {
    pub fn wants(&self, frame: u64) -> bool {
        self.ranges.iter().any(|r| (r.0..r.1).contains(&frame))
    }

    pub fn frame(&self, frame: u64, played: u64, shown: &[u32]) -> Result<(), String> {
        let bytes: Vec<u8> = shown.iter().flat_map(|p| p.to_le_bytes()).collect();
        for &(from, to) in self.ranges.iter().filter(|r| (r.0..r.1).contains(&frame)) {
            let name = self.dir.join(format!("{from}-{to}"));
            let err = |e: std::io::Error| format!("{}: {e}", name.display());
            if frame == from {
                std::fs::create_dir_all(&self.dir).map_err(err)?;
                std::fs::write(name.with_extension("samples"), played.to_string()).map_err(err)?;
            }
            std::fs::OpenOptions::new().create(true).write(true).append(frame != from).truncate(frame == from).open(name.with_extension("bgr0")).map_err(err)?.write_all(&bytes).map_err(err)?;
        }
        Ok(())
    }
}
