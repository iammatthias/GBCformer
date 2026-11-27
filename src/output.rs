//! Files a run writes: frames as PPM, audio as WAV, and the frame hash the gates compare.

use crate::machine::apu::{SAMPLE_RATE, Sample};
use crate::machine::ppu::{Frame, HEIGHT, WIDTH, rgb24};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// SHA-1 of the frame's RGB bytes, the `frame_sha1` the gates compare.
pub fn frame_sha1(frame: &Frame) -> String {
    sha1_smol::Sha1::from(rgb24(frame)).digest().to_string()
}

pub fn write_ppm(path: &Path, frame: &Frame) -> Result<(), String> {
    let mut out = format!("P6\n{WIDTH} {HEIGHT}\n255\n").into_bytes();
    out.extend(rgb24(frame));
    std::fs::write(path, out).map_err(|e| format!("{}: {e}", path.display()))
}

/// 16-bit PCM stereo at the APU's sample rate.
pub fn write_wav(path: &Path, samples: &[Sample]) -> Result<(), String> {
    let header = wav_header(samples.len() as u64)?;
    let io = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut out = BufWriter::new(File::create(path).map_err(io)?);
    out.write_all(&header).map_err(io)?;
    for v in samples.iter().flatten() {
        out.write_all(&v.to_le_bytes()).map_err(io)?;
    }
    out.flush().map_err(io)
}

/// The RIFF header for `samples` stereo samples. RIFF sizes are u32, so a WAV holds at most
/// 4 GiB, about 6.2 hours at 48 kHz.
fn wav_header(samples: u64) -> Result<[u8; 44], String> {
    let data_len = samples
        .checked_mul(4)
        .and_then(|n| u32::try_from(n + 36).ok())
        .map(|riff_len| riff_len - 36)
        .ok_or_else(|| format!("{samples} samples is more than a WAV file can hold (4 GiB); write a shorter run"))?;
    let mut out = Vec::with_capacity(44);
    out.extend(b"RIFF");
    out.extend((36 + data_len).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(2u16.to_le_bytes());
    out.extend(SAMPLE_RATE.to_le_bytes());
    out.extend((SAMPLE_RATE * 4).to_le_bytes());
    out.extend(4u16.to_le_bytes());
    out.extend(16u16.to_le_bytes());
    out.extend(b"data");
    out.extend(data_len.to_le_bytes());
    Ok(out.try_into().expect("44-byte header"))
}

/// Writes every `every`th frame to `dir` as frame_NNNNN.ppm.
#[derive(Clone)]
pub struct Recorder {
    pub dir: PathBuf,
    pub every: u64,
}

impl Recorder {
    /// Records frame number `n` if it is due. Errors are reported, not fatal.
    pub fn record(&self, n: u64, frame: &Frame) {
        if n % self.every != 0 {
            return;
        }
        let _ = std::fs::create_dir_all(&self.dir);
        if let Err(e) = write_ppm(&self.dir.join(format!("frame_{n:05}.ppm")), frame) {
            eprintln!("{e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_sizes_and_the_4_gib_limit() {
        let h = wav_header(3).unwrap();
        assert_eq!(&h[..12], b"RIFF\x30\x00\x00\x00WAVE");
        assert_eq!(&h[36..], b"data\x0c\x00\x00\x00");
        assert_eq!(h[24..28], 48_000u32.to_le_bytes());
        let most = (u32::MAX as u64 - 36) / 4;
        assert_eq!(wav_header(most).unwrap()[40..], ((most * 4) as u32).to_le_bytes());
        assert_eq!(
            wav_header(most + 1),
            Err("1073741815 samples is more than a WAV file can hold (4 GiB); write a shorter run".into())
        );
        assert!(wav_header(u64::MAX).is_err());
    }
}
