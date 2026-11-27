//! The xorshift stream the randomised tests draw from. Each test seeds its own.

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn byte(&mut self) -> u8 {
        self.next() as u8
    }
}
