//! Cartridge identity (FR01): the header fields the host acts on, and the ROM's sha1.

use crate::machine::cart::Kind;

/// The Pokémon Yellow release GBCformer was built and measured against.
pub const YELLOW_SHA1: &str = "cc7d03262ebfaf2f06772c1a480c7d9d5f4a38e1";

#[derive(Debug)]
pub struct Header {
    pub title: String,
    pub cgb_flag: u8,
    pub sgb_flag: u8,
    pub cartridge_type: u8,
    pub rom_size: u8,
    pub ram_size: u8,
    pub sha1: String,
}

impl Header {
    /// Header byte 0x143 bit 7: the game knows about CGB hardware.
    pub fn cgb_aware(&self) -> bool {
        self.cgb_flag & 0x80 != 0
    }

    pub fn is_known_yellow(&self) -> bool {
        self.sha1 == YELLOW_SHA1
    }

    /// The board, or why the host cannot emulate it.
    pub fn kind(&self) -> Result<Kind, String> {
        Kind::from_type(self.cartridge_type)
    }
}

pub fn identify(rom: &[u8]) -> Result<Header, String> {
    if rom.len() < 0x150 {
        return Err(format!("{} bytes is too short for a cartridge header", rom.len()));
    }
    let title = rom[0x134..0x143].iter().take_while(|&&b| b != 0).map(|&b| b as char).collect();
    Ok(Header {
        title,
        cgb_flag: rom[0x143],
        sgb_flag: rom[0x146],
        cartridge_type: rom[0x147],
        rom_size: rom[0x148],
        ram_size: rom[0x149],
        sha1: sha1_smol::Sha1::from(rom).digest().to_string(),
    })
}
