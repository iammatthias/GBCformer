//! Presentation: each pixel's colour to the screen's and the frames shown, with every
//! constant from the wiring's `Present` (the samples' filter is `transport::HighPass`).

use gbcformer::wiring::Present;

/// The frame being drawn, the frame the last frame signal finished, and the one shown.
pub struct Screen {
    p: Present,
    colours: bool,
    back: Vec<u32>,
    finished: Vec<u32>,
    shown: Vec<u32>,
    pending: bool,
}

impl Screen {
    /// `frames` holds the drawn frame then the shown one.
    pub fn new(p: &Present, colours: bool, frames: &[u32]) -> Screen {
        let n = p.width * p.height;
        assert_eq!(frames.len(), 2 * n, "the snapshot's frames");
        Screen { p: p.clone(), colours, back: frames[..n].to_vec(), finished: frames[..n].to_vec(), shown: frames[n..].to_vec(), pending: false }
    }

    fn colour(&self, c: u16) -> u32 {
        if !self.colours {
            return self.p.shades[c as usize & (self.p.shades.len() - 1)];
        }
        let mask = self.p.widen.len() as u16 - 1;
        self.p.channel_shifts.iter().fold(0u32, |rgb, &s| rgb << 8 | self.p.widen[(c >> s & mask) as usize] as u32)
    }

    /// Row `y` from each pixel's low and high byte.
    pub fn row(&mut self, y: usize, low: &[u8], high: &[u8]) {
        for x in 0..self.p.width {
            self.back[y * self.p.width + x] = self.colour(u16::from_le_bytes([low[x], high[x]]));
        }
    }

    pub fn end_frame(&mut self) {
        self.finished.copy_from_slice(&self.back);
        self.pending = true;
    }

    fn settle(&mut self) {
        if std::mem::take(&mut self.pending) {
            self.shown.copy_from_slice(&self.finished);
        }
    }

    /// The screen blanks; a frame finished before shows first.
    pub fn blank(&mut self) {
        self.settle();
        let c = if self.colours { self.p.blank_colour } else { self.p.blank_shade };
        self.shown.fill(c);
    }

    pub fn shown(&mut self) -> &[u32] {
        self.settle();
        &self.shown
    }
}

