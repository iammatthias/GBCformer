//! The cells' K/V rows behind their worker thread (`Cells::defer`): one stream of write
//! records leaves the same rows in every cell as it does when each record's rows are
//! placed at once, and the full scan over them reads every written line.

use gbcformer::engine::Engine;
use gbcformer::llama::Model;
use gbcformer::memory::attn::Params;
use gbcformer::memory::image::Image;
use gbcformer::memory::layout::{Banking, Mapper, LINE_BITS};
use gbcformer::sm83::build_mem;

fn engine() -> Engine {
    let mut e = Engine::new(Model::compile_mem(&build_mem(), Params::memory()));
    let banking = Banking::new(Mapper::Rom, 0x8000, 0x2000, false).unwrap();
    e.insert(Image::new(&[0; 0x8000], 0x2000, banking).with_writable_rom());
    e
}

/// Byte writes over 64 lines. First each line's byte 0 takes one of four values, so a
/// line returns to a line it held before: memo hits that change the cell, and more than
/// 64 of them, so some are built again and compared. Then fresh values at any byte: more
/// misses than the memo's first size holds, so it grows and its entries are replaced.
/// Then byte 0 again, over the lines the second run left.
fn stream() -> Vec<(u32, u8)> {
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut writes = Vec::new();
    for phase in 0..3 {
        for _ in 0..if phase == 1 { 6000 } else { 1500 } {
            let x = next();
            let addr = 0x4000 + (x >> 8) as u32 % 512;
            writes.push(if phase == 1 { (addr, (x >> 32) as u8) } else { (addr & !7, (x >> 32) as u8 & 3) });
        }
    }
    writes
}

#[test]
fn deferred_rows_are_the_rows_placed_at_once() {
    let (mut at_once, mut deferred) = (engine(), engine());
    deferred.memory.as_mut().unwrap().cells.defer();
    let writes = stream();
    for &(addr, byte) in &writes {
        at_once.keep_write(addr, byte);
        deferred.keep_write(addr, byte);
    }
    let (a, d) = (&at_once.memory.as_ref().unwrap().cells, &deferred.memory.as_ref().unwrap().cells);
    assert!(a.memo_checked > 16 && a.memo_len() > 1024, "the stream reaches memo checks ({}) and a grown memo ({} entries)", a.memo_checked, a.memo_len());
    assert_eq!((d.memo_hits, d.memo_checked, d.memo_len()), (a.memo_hits, a.memo_checked, a.memo_len()));
    let (rows_a, rows_d): (Vec<_>, Vec<_>) = (a.rows_by_line().collect(), d.rows_by_line().collect());
    assert_eq!(rows_a.len(), rows_d.len());
    for (x, y) in rows_a.iter().zip(&rows_d) {
        assert!(x == y, "the rows of line {:#07x} differ", x.0);
    }
    let layer = *d.builder.memory_layers.first().unwrap();
    let image = &deferred.memory.as_ref().unwrap().image;
    for n in (0x4000 >> LINE_BITS)..((0x4000 + 512) >> LINE_BITS) {
        assert_eq!(d.scan(layer, n), Some(image.line(n)), "the full scan of line {n:#07x}");
        assert_eq!(d.serve(layer, n).map(|r| r.0), Some(image.line(n)), "the lookup of line {n:#07x}");
    }
}
