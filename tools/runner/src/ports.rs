//! Port dispatch: a physical address the model emitted names a port block by its high
//! bits, by concatenation; the wiring says which stream each block reaches and what
//! address byte it takes there.

use gbcformer::wiring::{self, Target};

/// A port: the wiring's target and the address's low bits.
#[derive(Clone, Copy, Debug)]
pub struct Port {
    pub code: u32,
    pub low: u32,
}

pub struct Ports {
    w: wiring::Ports,
    by_code: Vec<Option<Target>>,
}

impl Ports {
    pub fn new(w: &wiring::Ports) -> Ports {
        let n = w.codes.iter().map(|(c, _)| *c as usize + 1).max().unwrap_or(0);
        let mut by_code = vec![None; n];
        for (c, t) in &w.codes {
            by_code[*c as usize] = Some(t.clone());
        }
        Ports { w: w.clone(), by_code }
    }

    /// The port `p` names, if any.
    pub fn port(&self, p: u32) -> Option<Port> {
        let block = 1u32 << self.w.low_bits;
        let code = p.checked_sub(self.w.base)? / block;
        self.by_code.get(code as usize)?.as_ref()?;
        Some(Port { code, low: p & (block - 1) })
    }

    pub fn target(&self, port: Port) -> &Target {
        self.by_code[port.code as usize].as_ref().expect("a port's target")
    }

    /// Whether a read of `port` is indirect (`wiring::Ports::indirect_reads`).
    pub fn indirect(&self, port: Port) -> bool {
        self.w.indirect_reads.iter().any(|(code, lows)| *code == port.code && lows.contains(&(port.low as u8)))
    }
}

/// What a stream that reads through ports needs from the others.
pub trait Access {
    fn read(&mut self, port: Port) -> u8;
    /// The spans stream's locks, a pass.
    fn lock_query(&mut self) -> u8;
}
