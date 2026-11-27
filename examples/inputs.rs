//! Each part's input slots as its layer-0 gathers write them: per slot, which of its
//! eight bit wires and two nibble wires hold a residual channel. With the parts' input
//! slots shared by index (`docs/perf-log.md`, "Round 13"), a slot's gather writes every
//! part's wires, so a part holds a foreign value in each channel another part reads at
//! that slot and it does not.

use gbcformer::one_model::{parts, Part};

fn main() {
    let ps = parts();
    // Per part and slot, a mask: bits 0..8 the bit wires, bits 8 and 9 the nibbles.
    // `Circuit::unread` marks head line bits only, so every input bit counts here whether
    // a node reads it or not: each holds its channel either way (`overlay_channels`).
    let used: Vec<Vec<u16>> = ps
        .iter()
        .map(|c| {
            let unread = c.unread();
            (0..c.inputs.len())
                .map(|j| {
                    let bits = (0..8).filter(|&i| !unread[c.inputs[j].1[i].0 as usize]).fold(0u16, |m, i| m | 1 << i);
                    c.values.iter().filter(|&&(s, _, w)| s == j && !unread[w.0 as usize]).fold(bits, |m, &(_, nib, _)| m | 1 << (8 + nib))
                })
                .collect()
        })
        .collect();
    let slots = used.iter().map(Vec::len).max().unwrap();
    let union: Vec<u16> = (0..slots).map(|j| used.iter().filter_map(|u| u.get(j)).fold(0, |m, &x| m | x)).collect();
    println!("shared by index: {slots} slots, {} channels in the union ({} bit wires, {} nibble wires)", union.iter().map(|m| m.count_ones()).sum::<u32>(), union.iter().map(|m| (m & 0xff).count_ones()).sum::<u32>(), union.iter().map(|m| (m >> 8).count_ones()).sum::<u32>());
    for (p, u) in Part::ALL.iter().zip(&used) {
        let own: u32 = u.iter().map(|m| m.count_ones()).sum();
        let foreign = |mask: u16| u.iter().zip(&union).map(|(m, un)| (un & !m & mask).count_ones()).sum::<u32>();
        println!("{:8} {:3} slots, {:3} channels of its own ({} nibble wires); foreign in its slots: {} bit wires, {} nibble wires", p.name(), u.len(), own, u.iter().map(|m| (m >> 8).count_ones()).sum::<u32>(), foreign(0xff), foreign(0x300));
    }
}
