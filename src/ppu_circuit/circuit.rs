//! One PPU scanline as a threshold circuit whose memory heads read VRAM, OAM and the
//! display bytes (`docs/ppu-in-weights.md`): the palettes and the display registers, which
//! the bus writes as records. Inputs are LY, the window's counter and FLG (with the window's
//! arming and carry); every output position is one byte of
//! one pixel's final colour, after the palette lookup: CGB's RGB555 (the low byte, or
//! the high one when FLG bit 2 asks) or DMG's 2-bit shade. The 160 pixel positions share
//! the per-pixel weights, and each reads its x from the position wires
//! (`Circuit::positions`). Two shared outputs carry the window's state to the next line.
//!
//! Every head reads one 8-byte line (`circuit::LINE`); the nodes that read a line pick
//! its byte. A constant-address line needs no pick; OAM line i, for example, holds
//! entries 2i and 2i + 1. A pixel's own heads ask position-dependent lines, which are its
//! tile's map, attribute and row lines and its palette line (four colours, two bytes
//! each).

use crate::circuit::{low_match, not, one, value, zero, Circuit, Lin, Line, OutKind, Query, Sig, Step, LINE, LINE_BITS};
use crate::machine::fifo::{self, Window};
use crate::machine::ppu::Mode;
use crate::memory::layout::{OAM, PAL, PAL_DMG, PAL_REGS, VRAM};

pub const WIDTH: usize = 160;
const SPRITES: usize = 40;
const SLOTS: usize = 10;

/// Input slots: LY, the window line counter, and FLG (bit 0 window armed, bit 1 CGB mode,
/// bit 2 the colour's high byte, bit 3 DMG compatibility mode, bit 4 ordinary STOP keeps
/// the PPU from VRAM, so every colour is 0, bit 5 the window carried from a DMG line that
/// ended at WX 166). The display registers are memory bytes (`regs`).
pub const INPUTS: [&str; 3] = ["LY", "WLN", "FLG"];
/// The window's state after the line: its line counter, and `WAR'` bit 0 armed, bit 1
/// carried into the next line.
pub const WINDOW: [&str; 2] = ["WLN'", "WAR'"];
/// The FLG and MFLG bit of the window carried into the line.
pub const CARRIED_IN: u8 = 0x20;
/// The `WAR'` and `MWAR'` bit of the window carried into the next line.
pub const CARRIED_OUT: u8 = 0x02;
/// The line's length (`fifo::length`: the dots to mode 0 less `lcd::LENGTH_BASE`), which the
/// control logic's timing takes to place mode 0.
pub const LENGTH: &str = "ML";

pub(crate) type W = Vec<Lin>;

pub(crate) fn w(b: [Sig; 8]) -> W {
    b.iter().map(|&s| Lin::from(s)).collect()
}
pub(crate) fn k(c: i64) -> Lin {
    Lin::konst(c)
}
pub(crate) fn nt(l: &Lin) -> Lin {
    one() - l.clone()
}
pub(crate) fn sum(ls: impl IntoIterator<Item = Lin>) -> Lin {
    ls.into_iter().fold(zero(), |a, l| a + l)
}
/// `l` when `v` is 1, `¬l` when 0.
pub(crate) fn lit(l: &Lin, v: usize) -> Lin {
    if v == 1 { l.clone() } else { nt(l) }
}

/// A conjunction of 0/1 literals. `ge` and `eq` fold a further threshold or equality
/// over an integer form into the same step by weighting the literals above the form's
/// headroom, as `sm83::translate::Cond` does.
#[derive(Clone)]
pub(crate) struct Cond {
    pub(crate) lin: Lin,
    pub(crate) n: i64,
}

impl Cond {
    pub(crate) fn all(lits: &[Lin]) -> Cond {
        Cond { lin: sum(lits.iter().cloned()), n: lits.len() as i64 }
    }
    /// The 0/1 forms `bits` spell `v` (`circuit::low_match`).
    pub(crate) fn spells(bits: &[Lin], v: usize) -> Cond {
        Cond { lin: low_match(bits, v), n: bits.len() as i64 }
    }
    pub(crate) fn and(mut self, l: &Lin) -> Cond {
        self.lin = self.lin + l.clone();
        self.n += 1;
        self
    }
    pub(crate) fn step(&self) -> Step {
        Step { coef: 1, lin: self.lin.clone(), theta: self.n }
    }
    /// `[cond ∧ lin ≥ v]`, with `hi` an upper bound of `lin`.
    pub(crate) fn ge(&self, lin: Lin, v: i64, hi: i64) -> Step {
        let b = (hi - v + 1).max(1);
        Step { coef: 1, lin: lin + self.lin.clone() * b, theta: v + b * self.n }
    }
    /// `[cond ∧ lin == v]` as two steps.
    pub(crate) fn eq(&self, lin: Lin, v: i64, hi: i64) -> [Step; 2] {
        let b = (hi - v + 1).max(1);
        let f = lin + self.lin.clone() * b;
        [Step { coef: 1, lin: f.clone(), theta: v + b * self.n }, Step { coef: -1, lin: f, theta: v + 1 + b * self.n }]
    }
}

/// `[lin == v]` as two steps.
pub(crate) fn eq2(lin: Lin, v: i64) -> Vec<Step> {
    vec![Step { coef: 1, lin: lin.clone(), theta: v }, Step { coef: -1, lin, theta: v + 1 }]
}

/// A VRAM query: `VRAM + bank·0x2000 + offset`, offset 13 bits.
pub(crate) fn vram_query(bank: Lin, off: &[Lin]) -> Query {
    assert_eq!(off.len(), 13);
    let mut q: Vec<Lin> = off.to_vec();
    q.push(bank);
    for b in 14..crate::circuit::ADDR_BITS {
        q.push(k(((VRAM >> b) & 1) as i64));
    }
    Query::Wires(q)
}

/// An OAM query: `OAM + offset`, offset 8 bits.
pub(crate) fn oam_query(off: &[Lin]) -> Query {
    assert_eq!(off.len(), 8);
    let mut q: Vec<Lin> = off.to_vec();
    for b in 8..crate::circuit::ADDR_BITS {
        q.push(k(((OAM >> b) & 1) as i64));
    }
    Query::Wires(q)
}

/// The low address bits of a line's first byte. A head's query with them zero names the
/// line, and its readers pick the byte.
pub(crate) fn line_base() -> Vec<Lin> {
    vec![zero(); LINE_BITS]
}

/// What a background or window fetch gives a position: its two colour bits as steps
/// (`B::fetch`) and its tile's attribute bits 0, 1, 2 (palette) and 7 (priority).
struct Fetched {
    colour: [Vec<Step>; 2],
    attr: [Lin; 4],
    /// The attributes read as 0 (`Glitch::off_attr`).
    attr_off: Lin,
}

/// A chosen object after the priority sort: its X >> 3, its `X & 7` one-hot, its
/// attribute bits, and its two row bytes.
struct Slot {
    coarse: W,
    xl: W,
    attr: W,
    data: [W; 2],
}

/// The OAM scan's slots (`B::scan`), and when asked whether the line fetches an object at
/// X 0 and one at X 1 (the CGB's WX glitch runs fetcher steps during their fetches).
pub(crate) struct Scan {
    pub(crate) valid: W,
    pub(crate) index: Vec<W>,
    pub(crate) xs: Vec<Line>,
    pub(crate) xsel: Vec<W>,
    pub(crate) x0: Option<[Lin; 2]>,
}

/// How the CGB's WX glitch leaves a fetch at a position (`fifo::length_rule::glitch`):
/// `off[p]` clears the fetched plane p and `off_attr` the attributes; each `extra[p]` is a
/// further source of plane p under its literals, the fetched row's plane or tile 0's.
struct Glitch {
    off: [Lin; 2],
    off_attr: Lin,
    extra: [Vec<(Vec<Lin>, Source)>; 2],
}

#[derive(Clone, Copy)]
enum Source {
    /// The fetched row's plane.
    Own(usize),
    /// Tile 0's row: a stale tile index, attributes 0.
    Tile0(usize),
}

pub(crate) struct B {
    pub(crate) c: Circuit,
    pub(crate) lcdc: W,
    pub(crate) cgb: Lin,
    /// BGP, OBP0 and OBP1 as the constant head reads them, for the DMG shade.
    pub(crate) pals: [W; 3],
    /// DMG compatibility mode (FLG bit 3): only then is the DMG shade looked up.
    pub(crate) compat: Lin,
}

impl B {
    /// A memory head at `layer`, or later when the query's line bits arrive later (an
    /// f16 split can add a level on the way to them).
    pub(crate) fn head(&mut self, name: &str, layer: u32, q: Query) -> Line {
        self.head_in(name, layer, layer, q)
    }

    /// A memory head on the layer in `lo..=hi` (or from the query's level on) that holds
    /// the fewest heads so far, so the flexible heads fill the layers the fixed ones
    /// leave and the widest layer's head count stays low.
    pub(crate) fn head_in(&mut self, name: &str, lo: u32, hi: u32, q: Query) -> Line {
        let ready = match &q {
            Query::Wires(bits) => self.c.level_of(&bits[LINE_BITS..]),
            Query::Const(_) => 0,
        };
        let lo = lo.max(ready);
        let count = |l: u32| self.c.heads.iter().filter(|h| h.layer == l).count();
        let layer = (lo..=hi.max(lo)).min_by_key(|&l| (count(l), l)).unwrap();
        self.c.head(name, layer, q)
    }

    /// Bit `bit` of byte k of line j for each value of the 0/1 forms `f`, as one node.
    /// `pick(fv)` names (j, k) for `f == fv`.
    pub(crate) fn pick_by(&mut self, name: &str, f: &[Lin], src: &[Line], pick: &dyn Fn(usize) -> (usize, usize), bit: usize) -> Lin {
        let steps = (0..1 << f.len()).map(|fv| {
            let (j, kk) = pick(fv);
            Cond::spells(f, fv).and(&Lin::from(src[j][kk][bit])).step()
        });
        self.c.node(name, steps.collect())
    }

    /// The fetcher at one position, as the hardware's does it for the tile under the
    /// pixel [P1 "Pixel FIFO"]: `sx` is the pixel's map x (column `sx >> 3`, pixel
    /// `sx & 7`) and `row` its map y, both position-dependent wires, so every head here
    /// reads a line of its own at each position. The map and attribute heads read the
    /// line of the map row holding the column, and the column's low bits pick the tile's
    /// bytes from it. The data head reads the tile's row, flipped and banked by its
    /// attribute, and the row's low bits, the pixel and the X flip pick the pixel's two
    /// bits. The colour bits come back as steps, each also requiring `gate`, so the
    /// caller sums the background's and the window's into one node per bit.
    fn fetch(&mut self, name: &str, gate: &[Lin], sx: &[Lin], row: &[Lin], map_bit: usize, glitch: &Glitch) -> Fetched {
        let mut off = line_base();
        off.extend(sx[6..8].iter().cloned());
        off.extend_from_slice(&row[3..8]);
        off.push(self.lcdc[map_bit].clone());
        off.push(one());
        off.push(one());
        let map = self.head_in(&format!("{name}m"), 2, 2, vram_query(zero(), &off));
        let attrs = self.head_in(&format!("{name}a"), 2, 2, vram_query(one(), &off));
        let col = &sx[3..6];
        let tile: W = (0..8).map(|i| self.pick_by(&format!("{name}.t{i}"), col, &[map], &|fv| (0, fv), i)).collect();
        let mut attr: W = vec![zero(); 8];
        for q in [0, 1, 2, 3, 5, 6, 7] {
            attr[q] = self.pick_by(&format!("{name}.a{q}"), col, &[attrs], &|fv| (0, fv), q);
        }
        // With LCDC.4 set the address is tile·16. With it clear the address is 0x1000 +
        // signed tile·16, which is tile·16 for tiles 128..255 and 0x1000 + tile·16 below,
        // so bit 11 is tile.7 either way and bit 12 is ¬LCDC.4 ∧ ¬tile.7.
        let b12 = self.c.and(&format!("{name}.b12"), &[nt(&self.lcdc[4]), nt(&tile[7])]);
        // Y flip makes row bit b XOR attr.6, two steps over the two wires.
        let flipped: W = (0..3)
            .map(|b| {
                let s = row[b].clone() + attr[6].clone();
                self.c.node(&format!("{name}.py{b}"), vec![Step { coef: 1, lin: s.clone(), theta: 1 }, Step { coef: -1, lin: s, theta: 2 }])
            })
            .collect();
        // The line holds rows `fy & !3` to `fy | 3` in both planes, row fy's plane at byte
        // 2·(fy & 3) + plane.
        let mut off = line_base();
        off.push(flipped[2].clone());
        off.extend(tile.iter().cloned());
        off.push(b12);
        let line = self.head_in(&format!("{name}d"), 1, 1, vram_query(attr[3].clone(), &off));
        // The row's low and high bytes, as the fetcher reads them, then the pixel's bit.
        // A new row changes the 16 row-byte nodes, not every pixel step.
        let data: [W; 2] = [0, 1].map(|plane| (0..8).map(|i| self.row_byte(&format!("{name}.d{plane}.{i}"), &line, &flipped[..2], plane, i)).collect());
        // Tile 0's row, which a glitched fetch with a stale tile index reads: tile 0 under
        // LCDC.4, bank 0, no flip.
        let stale = glitch.extra.iter().flatten().any(|(_, s)| matches!(s, Source::Tile0(_)));
        let tile0: Option<[W; 2]> = stale.then(|| {
            let mut off = line_base();
            off.push(row[2].clone());
            off.extend((0..8).map(|_| zero()));
            off.push(nt(&self.lcdc[4]));
            let line = self.head(&format!("{name}0"), self.c.level_of(&off), vram_query(zero(), &off));
            [0, 1].map(|plane| (0..8).map(|i| self.row_byte(&format!("{name}.z{plane}.{i}"), &line, &row[..2], plane, i)).collect())
        });
        let xf = attr[5].clone();
        let colour = [0, 1].map(|plane| {
            (0..8usize)
                .flat_map(|f| {
                    let mut lits: Vec<Lin> = gate.to_vec();
                    lits.extend((0..3).map(|i| lit(&sx[i], f >> i & 1)));
                    let own = |extra: &[Lin], q: usize| {
                        let at = |flip: bool, bit: usize| Cond::all(&[lits.clone(), extra.to_vec(), vec![if flip { xf.clone() } else { nt(&xf) }, data[q][bit].clone()]].concat()).step();
                        [at(false, 7 - f), at(true, f)]
                    };
                    let mut steps: Vec<Step> = own(&[nt(&glitch.off[plane])], plane).into();
                    for (extra, source) in &glitch.extra[plane] {
                        match *source {
                            Source::Own(q) => steps.extend(own(extra, q)),
                            Source::Tile0(q) => {
                                let row = &tile0.as_ref().expect("tile 0's row")[q];
                                steps.push(Cond::all(&[lits.clone(), extra.clone(), vec![row[7 - f].clone()]].concat()).step());
                            }
                        }
                    }
                    steps
                })
                .collect()
        });
        Fetched { colour, attr: [attr[0].clone(), attr[1].clone(), attr[2].clone(), attr[7].clone()], attr_off: glitch.off_attr.clone() }
    }

    /// Bit i of plane `plane` of the row whose bits 0, 1 are `row`, from a data line
    /// holding four rows.
    pub(crate) fn row_byte(&mut self, name: &str, line: &Line, row: &[Lin], plane: usize, i: usize) -> Lin {
        let steps = (0..4).map(|r| Cond::spells(row, r).and(&Lin::from(line[2 * r + plane][i])).step()).collect();
        self.c.node(name, steps)
    }

    /// The OAM scan: the first ten entries whose rows cover LY, in OAM order. `valid[s]`
    /// says slot `s` holds one, `index[s]` its OAM index bits, `xs[s]` the line of its
    /// entry pair and `xsel[s]` its X, which the mode-3 length reads.
    pub(crate) fn scan(&mut self, ly: &W, fetch: Option<Lin>, left: bool) -> Scan {
        let lines: Vec<Line> = (0..SPRITES / 2).map(|i| self.head(&format!("y{i}"), 1, Query::Const(OAM + (i * LINE) as u32))).collect();
        let ys: Vec<W> = (0..SPRITES).map(|e| w(lines[e / 2][4 * (e % 2)])).collect();
        let row = |y: &W| value(ly, 8) + 16 - value(y, 8);
        let vis: W = ys
            .iter()
            .enumerate()
            .map(|(i, y)| self.c.node(&format!("vis{i}"), vec![Step { coef: 1, lin: row(y), theta: 0 }, Step { coef: -1, lin: row(y) - self.lcdc[2].clone() * 8, theta: 8 }]))
            .collect();
        // CGB hardware fetches the objects with LCDC.1 clear too (they lengthen mode 3);
        // the pixel shows none of them then (`opaque`).
        let fetch = fetch.unwrap_or_else(|| self.c.or("objects fetched", &[self.lcdc[1].clone(), self.cgb.clone(), self.compat.clone()]));
        let chosen: W = (0..SPRITES).map(|i| self.c.ge(&format!("ch{i}"), vis[i].clone() * 10 - sum(vis[..i].iter().cloned()) + fetch.clone() * 10, 11)).collect();
        let count = sum(chosen.iter().cloned());
        let valid: W = (0..SLOTS).map(|s| self.c.ge(&format!("sv{s}"), count.clone(), s as i64 + 1)).collect();
        // The OAM scan's count [P1 "OAM"]. Slot s holds entry i when i is chosen with s
        // chosen before it, one node per (s, i). At most one entry holds a slot, so the
        // slot's OAM index bits are free sums of those nodes.
        let holds: Vec<Vec<(usize, Lin)>> = (0..SLOTS)
            .map(|s| (s..SPRITES).map(|i| (i, self.c.node(&format!("hold{s}.{i}"), Cond::all(&[chosen[i].clone()]).eq(sum(chosen[..i].iter().cloned()), s as i64, i as i64).to_vec()))).collect())
            .collect();
        let index: Vec<W> = holds.iter().map(|h| (0..6).map(|b| sum(h.iter().filter(|(i, _)| i >> b & 1 == 1).map(|(_, l)| l.clone()))).collect()).collect();
        // Index bits 1..6 name the line of an entry's pair, and index bit 0 picks the entry.
        let entry_line = |index: &W| {
            let mut off = line_base();
            off.extend(index[1..].iter().cloned());
            oam_query(&off)
        };
        let xs: Vec<Line> = (0..SLOTS).map(|s| self.head(&format!("sx{s}"), 4, entry_line(&index[s]))).collect();
        // Each slot's X, its entry picked by index bit 0, for the mode-3 length.
        let xsel: Vec<W> = (0..SLOTS)
            .map(|s| (0..8).map(|i| self.c.node(&format!("xo{s}.{i}"), (0..2).map(|e| Cond::all(&[lit(&index[s][0], e), Lin::from(xs[s][4 * e + 1][i])]).step()).collect())).collect())
            .collect();
        // An object the line fetches at X 0 or at X 1, from the entries' own X bytes.
        let at_x = |c: &mut Circuit, v: usize| {
            let terms: W = (0..SPRITES).map(|e| c.node(&format!("x{v}.{e}"), vec![Cond::all(&[vec![chosen[e].clone()], (0..8).map(|i| lit(&Lin::from(lines[e / 2][4 * (e % 2) + 1][i]), v >> i & 1)).collect()].concat()).step()])).collect();
            c.or(&format!("x{v}"), &terms)
        };
        let x0 = left.then(|| [at_x(&mut self.c, 0), at_x(&mut self.c, 1)]);
        Scan { valid, index, xs, xsel, x0 }
    }

    /// Each scan slot's place in the fetch order: the valid slots before it, by X and then
    /// OAM order, or in OAM order alone where `cgb` (CGB priority) is 1.
    pub(crate) fn ranks(&mut self, scan: &Scan, cgb: &Lin) -> W {
        let Scan { valid, index, xs, .. } = scan;
        let x_of = |s: usize, e: usize| value(&w(xs[s][4 * e + 1]), 8);
        // before[p][q] (p < q) is 1 when slot p comes before slot q: X_p ≤ X_q (the sort
        // is stable), or always under CGB priority. It is zero when slot q holds no
        // object. One step per pair of entries the two slots' index bit 0 can pick, over
        // that pair's X comparison, one level earlier.
        let before: Vec<Vec<Lin>> = (0..SLOTS)
            .map(|p| {
                (0..SLOTS)
                    .map(|q| {
                        if p >= q {
                            return zero();
                        }
                        let mut steps: Vec<Step> = (0..4)
                            .map(|ab| {
                                let (a, b) = (ab & 1, ab >> 1);
                                let le = self.c.ge(&format!("le{p}.{q}.{ab}"), x_of(q, b) - x_of(p, a), 0);
                                Cond::all(&[nt(cgb), lit(&index[p][0], a), lit(&index[q][0], b), valid[q].clone(), le]).step()
                            })
                            .collect();
                        steps.push(Cond::all(&[cgb.clone(), valid[q].clone()]).step());
                        self.c.node(&format!("bf{p}.{q}"), steps)
                    })
                    .collect()
            })
            .collect();
        (0..SLOTS).map(|s| sum((0..s).map(|p| before[p][s].clone())) + sum((s + 1..SLOTS).map(|q| valid[q].clone() - before[s][q].clone()))).collect()
    }

    /// Chooses the line's objects: the first ten whose rows cover LY, sorted by X then
    /// OAM index on DMG and by OAM index on CGB, each slot's OAM entry read through a
    /// head at the sorted index. `valid[s]` says slot `s` holds an object.
    fn sprites(&mut self, ly: &W) -> (Vec<Slot>, Vec<Lin>, Vec<W>, [Lin; 2]) {
        let scan = self.scan(ly, None, true);
        let cgb = self.cgb.clone();
        let rank = self.ranks(&scan, &cgb);
        let Scan { valid, index, xsel, x0, .. } = scan;
        let entry_line = |index: &W| {
            let mut off = line_base();
            off.extend(index[1..].iter().cloned());
            oam_query(&off)
        };
        let sorted: Vec<W> = (0..SLOTS)
            .map(|r| {
                (0..6)
                    .map(|b| {
                        let steps = (0..SLOTS).flat_map(|s| Cond::all(&[valid[s].clone(), index[s][b].clone()]).eq(rank[s].clone(), r as i64, SLOTS as i64 - 1)).collect();
                        self.c.node(&format!("so{r}.{b}"), steps)
                    })
                    .collect()
            })
            .collect();
        let slots = (0..SLOTS)
            .map(|s| {
                let line = self.head(&format!("o{s}"), 7, entry_line(&sorted[s]));
                // Entry e of the line is bytes 4e (Y), 4e + 1 (X), 4e + 2 (tile) and
                // 4e + 3 (attributes). The sorted index's bit 0 picks it in every node below.
                let s0 = sorted[s][0].clone();
                let byte = |e: usize, b: usize| w(line[4 * e + b]);
                let case = |e: usize| Cond::all(&[lit(&s0, e)]);
                let pick = |c: &mut Circuit, name: String, b: usize, i: usize| c.node(&name, (0..2).map(|e| case(e).and(&byte(e, b)[i]).step()).collect());
                // Row within the object, flipped by attr.6 over its 8 or 16 rows; the
                // tile byte's bit 0 gives way to the row's bit 3 for 8x16 objects.
                let low = |e: usize, b: usize| value(ly, b) + value(&not(&byte(e, 0)), b) + (17 & ((1 << b) - 1));
                let rows: W = (0..3)
                    .map(|b| {
                        let steps = (0..2).flat_map(|e| self.bit_steps(&case(e), low(e, b + 1) + byte(e, 3)[6].clone() * (1 << b), b as u32)).collect();
                        self.c.node(&format!("sr{s}.{b}"), steps)
                    })
                    .collect();
                let mut steps: Vec<Step> = Vec::new();
                for e in 0..2 {
                    let f = low(e, 4) + byte(e, 3)[6].clone() * 8;
                    let tall = case(e).and(&self.lcdc[2]);
                    steps.extend((1..=4).map(|j| {
                        let mut st = tall.ge(f.clone(), 8 * j, 39);
                        st.coef = if j % 2 == 1 { 1 } else { -1 };
                        st
                    }));
                    steps.push(case(e).and(&nt(&self.lcdc[2])).and(&byte(e, 2)[0]).step());
                }
                let b4 = self.c.node(&format!("sr{s}.3"), steps);
                let bank = self.c.node(&format!("sb{s}"), (0..2).map(|e| case(e).and(&self.cgb).and(&byte(e, 3)[3]).step()).collect());
                let tile: W = (1..8).map(|i| pick(&mut self.c, format!("st{s}.{i}"), 2, i)).collect();
                let coarse: W = (3..8).map(|i| pick(&mut self.c, format!("sx{s}.{i}"), 1, i)).collect();
                let mut attr: W = vec![zero(); 8];
                for q in [0, 1, 2, 4, 5, 7] {
                    attr[q] = pick(&mut self.c, format!("sa{s}.{q}"), 3, q);
                }
                let xl: W = (0..8).map(|v| self.c.node(&format!("xl{s}={v}"), (0..2).flat_map(|e| case(e).eq(value(&byte(e, 1), 3), v, 7)).collect())).collect();
                let mut off = line_base();
                off.push(rows[2].clone());
                off.push(b4);
                off.extend(tile);
                off.push(zero());
                let data_line = self.head(&format!("od{s}"), 8, vram_query(bank, &off));
                let data = [0, 1].map(|plane| (0..8).map(|i| self.row_byte(&format!("od{s}.{plane}.{i}"), &data_line, &rows[..2], plane, i)).collect());
                Slot { coarse, xl, attr, data }
            })
            .collect();
        (slots, valid, xsel, x0.expect("the objects at X 0 and 1"))
    }

    /// The steps of bit `k` of the non-negative form `s` when `cond` holds (0 when it
    /// does not). They are `Circuit::bit_of`'s staircase with each step gated.
    pub(crate) fn bit_steps(&self, cond: &Cond, s: Lin, k: u32) -> Vec<Step> {
        let p = 1i64 << k;
        assert!(self.c.bounds(&s).0 >= 0, "bit_steps needs a non-negative form");
        let s = self.c.modulo(s, 2 * p);
        let hi = self.c.bounds(&s).1;
        (1..=hi / p)
            .map(|j| {
                let mut st = cond.ge(s.clone(), j * p, hi);
                st.coef = if j % 2 == 1 { 1 } else { -1 };
                st
            })
            .collect()
    }

    /// The fields of the pixel that shows, from its background pixel (colour bits,
    /// palette bits, priority attribute) and object pixel (colour, palette, priority,
    /// present). Each is the sum, over the exclusive cases where a layer shows, of that
    /// case with the layer's field. They are the palette read's address
    /// (`Shown::query`) and the colour's byte pick.
    fn shown(&mut self, bg: &[Lin; 6], obj: &[Lin; 7]) -> Shown {
        let [bc0, bc1, bp0, bp1, bp2, bprio] = bg;
        let [oc0, oc1, op0, op1, op2, oprio, present] = obj;
        let (lcdc0, cgb) = (self.lcdc[0].clone(), self.cgb.clone());
        let bg_zero = [nt(bc0), nt(bc1)];
        // Exclusive cases of "the object shows" and of "the background shows".
        let object_shows = vec![
            Cond::all(&[present.clone(), nt(&cgb), nt(oprio)]),
            Cond::all(&[present.clone(), nt(&cgb), oprio.clone(), bg_zero[0].clone(), bg_zero[1].clone()]),
            Cond::all(&[present.clone(), cgb.clone(), nt(&lcdc0)]),
            Cond::all(&[present.clone(), cgb.clone(), lcdc0.clone(), bg_zero[0].clone(), bg_zero[1].clone()]),
            Cond::all(&[present.clone(), cgb.clone(), lcdc0.clone(), bc0.clone(), nt(bprio), nt(oprio)]),
            Cond::all(&[present.clone(), cgb.clone(), lcdc0.clone(), nt(bc0), bc1.clone(), nt(bprio), nt(oprio)]),
        ];
        let background_shows = vec![
            Cond::all(&[nt(present)]),
            Cond::all(&[present.clone(), nt(&cgb), oprio.clone(), bc0.clone()]),
            Cond::all(&[present.clone(), nt(&cgb), oprio.clone(), nt(bc0), bc1.clone()]),
            Cond::all(&[present.clone(), cgb.clone(), lcdc0.clone(), bc0.clone(), bprio.clone()]),
            Cond::all(&[present.clone(), cgb.clone(), lcdc0.clone(), nt(bc0), bc1.clone(), bprio.clone()]),
            Cond::all(&[present.clone(), cgb.clone(), lcdc0.clone(), bc0.clone(), nt(bprio), oprio.clone()]),
            Cond::all(&[present.clone(), cgb.clone(), lcdc0.clone(), nt(bc0), bc1.clone(), nt(bprio), oprio.clone()]),
        ];
        // A field over both layers' cases, each case also gated by `gate` when given. It is
        // the object's field `of` where an object shows and the background's `bf` where
        // the background does.
        let field = |c: &mut Circuit, name: &str, gate: Option<&Lin>, of: Option<&Lin>, bf: Option<&Lin>| -> Lin {
            let gated = |cs: &Cond| gate.map_or(cs.clone(), |g| cs.clone().and(g));
            let mut steps: Vec<Step> = Vec::new();
            if let Some(f) = of {
                steps.extend(object_shows.iter().map(|cs| gated(cs).and(f).step()));
            }
            if let Some(f) = bf {
                steps.extend(background_shows.iter().map(|cs| gated(cs).and(f).step()));
            }
            c.node(name, steps)
        };
        let colour = [field(&mut self.c, "shown.c0", None, Some(oc0), Some(bc0)), field(&mut self.c, "shown.c1", None, Some(oc1), Some(bc1))];
        let line = [
            field(&mut self.c, "shown.p0", Some(&cgb), Some(op0), Some(bp0)),
            field(&mut self.c, "shown.p1", Some(&cgb), Some(op1), Some(bp1)),
            field(&mut self.c, "shown.p2", Some(&cgb), Some(op2), Some(bp2)),
            field(&mut self.c, "shown.obj", Some(&cgb), Some(&one()), None),
        ];
        // On DMG, which of BGP, OBP0 and OBP1 the pixel reads (an object's palette bit 0
        // selects OBP1 there).
        let dmg_obj = field(&mut self.c, "shown.dobj", Some(&nt(&cgb)), Some(&one()), None);
        let dmg_obp1 = field(&mut self.c, "shown.dobp1", Some(&nt(&cgb)), Some(op0), None);
        // The DMG shade of the pixel that shows, beside its colour: the 2 bits BGP, OBP0 or
        // OBP1 give its colour, in the cases where DMG rendering shows a layer. Only
        // compatibility mode reads it, so it is 0 outside it and holds still on a CGB line.
        let (pals, compat) = (self.pals.clone(), self.compat.clone());
        let dmg_shade: [Lin; 2] = std::array::from_fn(|j| {
            let mut steps = Vec::new();
            for c in 0..4usize {
                for cs in &object_shows[..2] {
                    for (r, pal) in pals[1..].iter().enumerate() {
                        steps.push(cs.clone().and(&compat).and(&lit(oc0, c & 1)).and(&lit(oc1, c >> 1)).and(&lit(op0, r)).and(&pal[2 * c + j]).step());
                    }
                }
                for cs in &background_shows[..3] {
                    steps.push(cs.clone().and(&compat).and(&lit(bc0, c & 1)).and(&lit(bc1, c >> 1)).and(&pals[0][2 * c + j]).step());
                }
            }
            self.c.node(&format!("dshade.{j}"), steps)
        });
        Shown { colour, line, dmg_obj, dmg_obp1, dmg_shade }
    }
}

/// The pixel that shows. It holds its colour and, on CGB, its palette line in palette RAM
/// (palette bits, then the object bit). The line is zero on DMG, where the object bits
/// pick BGP, OBP0 or OBP1.
struct Shown {
    colour: [Lin; 2],
    line: [Lin; 4],
    dmg_obj: Lin,
    dmg_obp1: Lin,
    /// DMG rendering's shade, which compatibility mode looks up in CGB palette RAM.
    dmg_shade: [Lin; 2],
}

impl Shown {
    /// The palette line the pixel's colour sits in: CGB's palette RAM line
    /// `obj·8 + palette` (four colours, two bytes each), or on DMG the line holding BGP,
    /// OBP0 and OBP1.
    fn query(&self, cgb: &Lin) -> Query {
        const { assert!(PAL % 256 == 0 && PAL_DMG == 128) };
        let mut q = line_base();
        q.extend(self.line.iter().cloned());
        q.push(nt(cgb));
        for b in 8..crate::circuit::ADDR_BITS {
            q.push(k(((PAL >> b) & 1) as i64));
        }
        Query::Wires(q)
    }
}

/// The position forms of the `Pos` shape. x as a value is affine in the position wires,
/// so it costs no level, and a map x's bits (`map_x`) cost one level.
struct At {
    p: Vec<Sig>,
}

impl At {
    /// The bits of `base + x + add - shift` mod 256 at each position, `shift` a 0/1 wire of
    /// the position. Bit n is the staircase of `base mod 2^(n+1)` plus the position's
    /// `(x + add) mod 2^(n+1)` less `shift`, plus 2^(n+1) to keep it positive: a sum below
    /// 3·2^(n+1), so it takes five steps, and its weights stay under f16's 511. The position's
    /// part is one weight per position wire.
    fn map_x(&self, c: &mut Circuit, name: &str, base: &[Lin], add: usize, shift: &Lin) -> W {
        (0..8)
            .map(|n| {
                let m = 1i64 << (n + 1);
                let s = value(base, n + 1) + sum((0..WIDTH).map(|x| Lin::from(self.p[x]) * ((x as i64 + add as i64) % m))) - shift.clone() + m;
                let p = m / 2;
                c.node(&format!("{name}.{n}"), (1..=5).map(|j| Step { coef: if j % 2 == 1 { 1 } else { -1 }, lin: s.clone(), theta: j * p }).collect())
            })
            .collect()
    }
    fn x(&self) -> Lin {
        sum((0..WIDTH).map(|x| Lin::from(self.p[x]) * x as i64))
    }
}

impl B {
    /// The pixel that shows at each output position, from the background's or the
    /// window's fetch (`fetch`, gated by `[bgon, ¬winx]` and `[bgon, winx]`) and from the
    /// objects, with the column and the fine position read from the position wires.
    #[allow(clippy::too_many_arguments)]
    fn pixel_at(&mut self, at: &At, bg: &Fetched, wn: &Fetched, winx: &Lin, bgon: &Lin, slots: &[Slot], valid: &[Lin]) -> Shown {
        let cgb = self.cgb.clone();
        let bgc: [Lin; 2] = [0, 1].map(|bit| self.c.node(&format!("bc.{bit}"), [bg.colour[bit].clone(), wn.colour[bit].clone()].concat()));
        let bga: Vec<Lin> = (0..4)
            .map(|q| self.c.node(&format!("ba.{q}"), vec![Cond::all(&[bgon.clone(), nt(winx), nt(&bg.attr_off), bg.attr[q].clone()]).step(), Cond::all(&[bgon.clone(), winx.clone(), nt(&wn.attr_off), wn.attr[q].clone()]).step()]))
            .collect();
        // The OBJ FIFO at this pixel [P1 "Pixel FIFO"]. Slot s covers x at its column
        // v = x + 8 − X when 0 ≤ v < 8, one node per v (X is 8·(X >> 3) plus the `X & 7`
        // one-hot's value).
        let hit: Vec<Vec<Lin>> = slots
            .iter()
            .enumerate()
            .map(|(s, sl)| {
                let x = value(&sl.coarse, 5) * 8 + sum((0..8).map(|k| sl.xl[k].clone() * k as i64));
                (0..8).map(|v| self.c.node(&format!("hit{s}.{v}"), eq2(x.clone() - at.x() + (v as i64 - 8), 0))).collect()
            })
            .collect();
        // Planes 0 and 1 give the slot's colour bit at this pixel, column v of its row
        // (bit 7 − v, or v when flipped). Plane 2 gives whether that pixel is opaque.
        let placed = |s: usize, lits: &[Lin], plane: usize| -> Vec<Step> {
            let sl = &slots[s];
            let flip = sl.attr[5].clone();
            (0..8usize)
                .flat_map(|v| {
                    [(nt(&flip), 7 - v), (flip.clone(), v)].map(|(fl, i)| {
                        let c = Cond::all(&[lits.to_vec(), vec![hit[s][v].clone(), fl]].concat());
                        if plane == 2 { c.ge(sl.data[0][i].clone() + sl.data[1][i].clone(), 1, 2) } else { c.and(&sl.data[plane][i]).step() }
                    })
                })
                .collect()
        };
        let opaque: W = (0..SLOTS).map(|s| self.c.node(&format!("op{s}"), placed(s, &[valid[s].clone(), self.lcdc[1].clone()], 2))).collect();
        // Slot s wins when it is opaque here and no earlier slot is. That is a conjunction
        // of literals, so each step below carries it instead of reading a level of its own.
        let wins = |s: usize| -> Vec<Lin> { std::iter::once(opaque[s].clone()).chain(opaque[..s].iter().map(nt)).collect() };
        let oc: [Lin; 2] = [0, 1].map(|bit| {
            let steps = (0..SLOTS).flat_map(|s| placed(s, &wins(s), bit)).collect();
            self.c.node(&format!("oc.{bit}"), steps)
        });
        let op: Vec<Lin> = (0..3)
            .map(|q| {
                let mut steps: Vec<Step> = (0..SLOTS).map(|s| Cond::all(&[wins(s), vec![cgb.clone(), slots[s].attr[q].clone()]].concat()).step()).collect();
                if q == 0 {
                    steps.extend((0..SLOTS).map(|s| Cond::all(&[wins(s), vec![nt(&cgb), slots[s].attr[4].clone()]].concat()).step()));
                }
                self.c.node(&format!("opl.{q}"), steps)
            })
            .collect();
        let oprio = self.c.node("opr", (0..SLOTS).map(|s| Cond::all(&[wins(s), vec![slots[s].attr[7].clone()]].concat()).step()).collect());
        let present = self.c.or("present", &opaque);
        let bgpx = [bgc[0].clone(), bgc[1].clone(), bga[0].clone(), bga[1].clone(), bga[2].clone(), bga[3].clone()];
        let objpx = [oc[0].clone(), oc[1].clone(), op[0].clone(), op[1].clone(), op[2].clone(), oprio, present];
        self.shown(&bgpx, &objpx)
    }
}

/// The display registers from `layout::PAL_REGS` on (LCDC, SCY, SCX, WY, WX), from a head
/// at layer 1 on the line of `PAL + PAL_DMG`, which holds BGP, OBP0, OBP1 and then them.
pub(crate) fn regs(c: &mut Circuit) -> [W; 8] {
    const { assert!(PAL_REGS == PAL_DMG + 3 && PAL_DMG as usize % LINE == 0) };
    let line = c.head("regs", 1, Query::Const(PAL + PAL_DMG));
    std::array::from_fn(|i| w(line[i]))
}

/// The window's cases on a quiet line (`fifo::length_rule::cases`), as wires. `armed` and
/// `lyeqwy` make SameBoy's `wy_triggered` as two exclusive cases (`armed_steps`), so a
/// condition on it stays one level.
struct Win {
    first: Lin,
    armed: Lin,
    lyeqwy: Lin,
    /// The window carried from a DMG line that ended at WX 166 (FLG bit 5), never on line 0.
    carried: Lin,
    /// SCX's fine scroll is not 0.
    fnz: Lin,
    /// WX 0 under a fine scroll.
    wx0f: Lin,
    wx247: Lin,
    /// The window starts at the line's check: enabled, armed, not carried, WX up to 165, or
    /// 166 on CGB hardware.
    start: Lin,
    /// The window starts at WX 0 under a fine scroll, a dot late.
    wx0start: Lin,
    /// The window is fetched from the line's first tile: carried and enabled, or started at
    /// WX 0 under a fine scroll.
    from_start: Lin,
    /// The window starts at WX − 7 (`start` and not `from_start`).
    std: Lin,
    /// A window line inserts a pixel where a pushed row meets WX (`insert_bg_pixel`).
    in_row: Lin,
    /// Issue 278: a DMG armed with the window disabled inserts one where a push meets WX.
    i278: Lin,
    /// A carried line inserts one at its junk row when WX is 247 (`P = -f`).
    i247: Lin,
    /// The rows the line counts: a start, a DMG's count at WX 166, the carry.
    counted: Lin,
    carry: Lin,
    armed_out: Lin,
}

impl Win {
    /// `lits ∧ wy_triggered` as two steps.
    fn armed_steps(&self, lits: &[Lin]) -> Vec<Step> {
        self.armed_cases(lits).iter().map(|l| Cond::all(l).step()).collect()
    }

    fn armed_cases(&self, lits: &[Lin]) -> Vec<Vec<Lin>> {
        vec![[lits.to_vec(), vec![self.armed.clone()]].concat(), [lits.to_vec(), vec![nt(&self.armed), self.lyeqwy.clone()]].concat()]
    }
}

/// `WAR'` and `MWAR'`: bit 0 the window armed, bit 1 carried into the next line ([`CARRIED_OUT`]).
fn window_out(win: &Win) -> [Lin; 8] {
    const { assert!(CARRIED_OUT == 2) };
    std::array::from_fn(|i| match i {
        0 => win.armed_out.clone(),
        1 => win.carry.clone(),
        _ => zero(),
    })
}

/// The window's cases from LY, FLG and the display registers. Each rule is `fifo::State`'s
/// on a line with no write (`window`, `push`, `window_after`); `fifo::length_rule` holds the
/// closed forms to `fifo::draw`.
fn window_cases(b: &mut B, ly: &W, flg: &W, scx: &W, wy: &W, wx: &W) -> Win {
    let first = b.c.and("LY=0", &(0..8).map(|i| nt(&ly[i])).collect::<Vec<_>>());
    let armed = b.c.and("armed", &[flg[0].clone(), nt(&first)]);
    let carried = b.c.and("carried", &[flg[5].clone(), nt(&first)]);
    let lyeqwy = b.c.node("lyeqwy", eq2(value(ly, 8) - value(wy, 8), 0));
    let en = b.lcdc[5].clone();
    let cgbhw = flg[1].clone() + flg[3].clone();
    let dmg = nt(&cgbhw);
    let f = value(scx, 3);
    let fnz = b.c.ge("f>0", f.clone(), 1);
    let wx0f = b.c.node("wx0f", vec![Cond::all(&(0..8).map(|i| nt(&wx[i])).collect::<Vec<_>>()).ge(f.clone(), 1, 7)]);
    let wxok = b.c.ge("wxok", k(165) + cgbhw.clone() - value(wx, 8), 0);
    let wx166 = b.c.node("wx=166", eq2(value(wx, 8), 166));
    let wx247 = b.c.node("wx=247", eq2(value(wx, 8), 247));
    let push7 = b.c.node("push7", eq2(value(wx, 3) + f.clone(), 7));
    let mut w = Win {
        first,
        armed,
        lyeqwy,
        carried: carried.clone(),
        fnz,
        wx0f: wx0f.clone(),
        wx247: wx247.clone(),
        start: zero(),
        wx0start: zero(),
        from_start: zero(),
        std: zero(),
        in_row: zero(),
        i278: zero(),
        i247: zero(),
        counted: zero(),
        carry: zero(),
        armed_out: zero(),
    };
    let nc = nt(&carried);
    w.start = b.c.node("wstart", w.armed_steps(&[nc.clone(), en.clone(), wxok.clone()]));
    w.wx0start = b.c.node("wx0start", w.armed_steps(&[nc.clone(), en.clone(), wx0f.clone()]));
    w.from_start = b.c.node("wfirst", [vec![Cond::all(&[carried.clone(), en.clone()]).step()], w.armed_steps(&[nc.clone(), en.clone(), wx0f.clone()])].concat());
    w.std = b.c.node("wstd", w.armed_steps(&[nc.clone(), en.clone(), wxok, nt(&wx0f)]));
    let in_row: Vec<Vec<Lin>> = [vec![vec![carried.clone(), en.clone(), push7.clone()]], w.armed_cases(&[nc.clone(), en.clone(), wx0f, push7.clone()])].concat();
    let i278 = w.armed_cases(&[dmg.clone(), nt(&en), push7]);
    w.in_row = b.c.node("in row", in_row.iter().map(|l| Cond::all(l).step()).collect());
    w.i278 = b.c.node("i278", i278.iter().map(|l| Cond::all(l).step()).collect());
    w.i247 = b.c.and("i247", &[carried, wx247]);
    let count166 = b.c.node("count166", w.armed_steps(&[nc, en.clone(), wx166.clone(), dmg.clone()]));
    w.carry = b.c.node("carry", w.armed_steps(&[dmg, en, wx166]));
    w.counted = w.start.clone() + count166 + w.carry.clone();
    w.armed_out = b.c.or("WAR'", &[w.armed.clone(), w.lyeqwy.clone()]);
    w
}

/// The scanline circuit: outputs `PX0..PX159` (one per pixel position, all the same
/// wires), the window's state after the line (`WINDOW`), then the write-record slots the
/// memory compiler expects, which a scanline leaves zero.
pub fn build() -> Circuit {
    let mut c = Circuit::default();
    let ly = w(c.input("LY"));
    let wln_in = w(c.input("WLN"));
    let flg = w(c.input("FLG"));
    // LCDC, SCY, SCX, WY and WX as of this line: the newest records on BGP's line, bytes
    // 3 to 7, which a constant head reads as the palette RAM reads a write the bus made.
    // Bytes 0 to 2 are BGP, OBP0 and OBP1, which compatibility mode's shade reads early.
    let [bgp, obp0, obp1, lcdc, scy, scx, wy, wx] = regs(&mut c);
    let (cgb, half) = (flg[1].clone(), flg[2].clone());
    let mut b = B { c, lcdc: lcdc.clone(), cgb: cgb.clone(), pals: [bgp, obp0, obp1], compat: flg[3].clone() };
    let at = At { p: b.c.positions(WIDTH) };

    // The window's state starts over with each frame, because the PPU draws line 0
    // first after VBlank and after the LCD turns on.
    let win = window_cases(&mut b, &ly, &flg, &scx, &wy, &wx);
    let wln: W = (0..8).map(|i| b.c.and(&format!("wln.{i}"), &[wln_in[i].clone(), nt(&win.first)])).collect();
    // DMG: LCDC.0 clear blanks background and window; CGB draws them regardless.
    let bgon = b.c.or("bgon", &[lcdc[0].clone(), cgb.clone()]);

    let (slots, valid, xsel, [x0, x1]) = b.sprites(&ly);
    let bgrow = b.c.sum_bits("bgrow", 8, |n| value(&scy, n) + value(&ly, n));
    // A carried line draws the row its carry counted, one before the counter.
    let wrow = b.c.sum_bits("wrow", 8, |n| value(&wln, n) + win.carried.clone() * ((1 << n) - 1));

    // The inserted pixel at P and the shift after it (`fifo::length_rule::cases`): at
    // `P = WX - 7` where a pushed row meets `pos + 7 == WX` (rows are pushed at `-f + 8n`, so
    // `(WX & 7) + f == 7`), for issue 278 and a window line's `insert_bg_pixel`; and a carried
    // line's junk row at WX 247, `P = -f`. Each case is one literal of its step, with `x > P`
    // or `x == P` at the position, so the step's weights fit f16.
    let (x, f) = (at.x(), value(&scx, 3));
    let after = x.clone() + 6 - value(&wx, 8);
    // Each case is its own node of one step, or of one equality: a step over the position
    // wires is reoriented (`Circuit::node_at`) and leaves its wire offset by its coefficient,
    // and a reused channel's clearing unit undoes an offset of 0 or −1 only.
    let ins = win.in_row.clone() + win.i278.clone();
    let junk = [win.carried.clone(), win.wx247.clone()];
    let shift = b.c.node("shift", vec![Cond::all(&[ins.clone()]).ge(after.clone(), 0, 165)]) + b.c.node("shift junk", vec![Cond::all(&junk).ge(x.clone() + win.fnz.clone(), 1, 160)]);
    let inserted = b.c.node("inserted", Cond::all(&[ins]).eq(after.clone() + 1, 0, 166).to_vec()) + b.c.node("inserted junk", vec![Cond::all(&[junk.to_vec(), vec![nt(&win.fnz)]].concat()).ge(-x.clone(), 0, 0)]);

    // The window's map x is x − (WX − 7) = x + ¬WX + 8 on a line that starts it, and on one
    // fetched from its first tile (carried, or WX 0 under a fine scroll) x + 8 + f, both less
    // the shift: a base of ¬WX or f. A carried line takes f, and so does WX 0 with a fine
    // scroll; ¬WX is 255 there.
    let base: W = (0..8)
        .map(|j| {
            let mut steps = vec![Cond::all(&[nt(&win.carried), nt(&win.wx0f), nt(&wx[j])]).step()];
            if j < 3 {
                steps.push(Cond::all(&[win.carried.clone(), scx[j].clone()]).step());
                steps.push(Cond::all(&[nt(&win.carried), win.wx0f.clone(), scx[j].clone()]).step());
            }
            b.c.node(&format!("wbase.{j}"), steps)
        })
        .collect();
    // The window shows from its start on a line that starts it, and everywhere on one
    // fetched from its first tile.
    let winx = b.c.node("wx", vec![Cond::all(&[win.from_start.clone()]).step(), Cond::all(&[win.std.clone()]).ge(x.clone() + 7 - value(&wx, 8), 0, 166)]);
    let bsx = at.map_x(&mut b.c, "bsx", &scx, 0, &shift);
    let wsx = at.map_x(&mut b.c, "wsx", &base, 8, &shift);

    // The CGB's WX glitch (`fifo::length_rule::glitch`), on CGB hardware with the window
    // enabled and armed. On a line fetched from its first tile at WX 0, window column tile
    // 1 (and tile 2 at a fine scroll of 7 with an object at X 1) has a stale tile index 0
    // and attributes 0; its planes survive only with no object at X 0 or 1, the low one at
    // a fine scroll of 1, the high one up to 3. At WX 247 to 252 the first background tile
    // is glitched as one T1 step met `pos + 7 == WX`.
    let cgbhw = flg[1].clone() + flg[3].clone();
    let glitchy = |b: &mut B, name: &str, lits: &[Lin]| b.c.node(name, win.armed_steps(&[vec![cgbhw.clone(), lcdc[5].clone()], lits.to_vec()].concat()));
    let fge = |b: &mut B, v: i64| b.c.ge(&format!("f>={v}"), f.clone(), v);
    let (f3, f5) = (fge(&mut b, 3), fge(&mut b, 5));
    let wxeq = |b: &mut B, v: i64| b.c.node(&format!("wx={v}"), eq2(value(&wx, 8), v));
    let early = [nt(&x0), nt(&x1)];
    let g247 = glitchy(&mut b, "g247", &[win.wx247.clone(), x0.clone()]);
    let wx248 = wxeq(&mut b, 248);
    let g248 = glitchy(&mut b, "g248", &[wx248, nt(&x0), win.fnz.clone()]);
    let wx250 = wxeq(&mut b, 250);
    let g250 = glitchy(&mut b, "g250", &[wx250, nt(&x0), f3]);
    let wx252 = wxeq(&mut b, 252);
    let g252 = glitchy(&mut b, "g252", &[wx252, nt(&x0), f5]);
    let first_tile = b.c.ge("bg first tile", k(7) - f.clone() - x.clone(), 0);
    let bfirst = |b: &mut B, name: &str, g: &[&Lin]| b.c.node(name, g.iter().map(|g| Cond::all(&[first_tile.clone(), (*g).clone()]).step()).collect());
    let stale = bfirst(&mut b, "bg stale", &[&g247, &g248]);
    let off0 = bfirst(&mut b, "bg off0", &[&g247, &g248, &g250]);
    let off1 = bfirst(&mut b, "bg off1", &[&g247, &g248, &g252]);
    let keep = vec![first_tile.clone(), g248.clone()];
    let bg_glitch = Glitch {
        off: [off0, off1],
        off_attr: stale + inserted.clone(),
        extra: [vec![(keep.clone(), Source::Tile0(0))], vec![(keep, Source::Tile0(1)), (vec![first_tile.clone(), g252.clone()], Source::Own(0))]],
    };
    let wg = glitchy(&mut b, "wg", &[nt(&win.carried), win.wx0f.clone()]);
    let w2 = glitchy(&mut b, "wg2", &[nt(&win.carried), win.wx0f.clone(), scx[0].clone(), scx[1].clone(), scx[2].clone(), x1.clone()]);
    let keep_low = b.c.node("wg keep0", vec![Cond::spells(&scx[..3], 1).and(&early[0]).and(&early[1]).step()]);
    let keep_high = b.c.node("wg keep1", vec![Cond::all(&early).ge(k(3) - f.clone(), 0, 3)]);
    // Window column tile 1 is wsx 8 to 15, tile 2 wsx 16 to 23.
    let tile_lits = |t: usize| -> Vec<Lin> { (3..8).map(|i| lit(&wsx[i], (t >> (i - 3)) & 1)).collect() };
    let w1 = [tile_lits(1), vec![wg.clone()]].concat();
    let wtile = b.c.node("wg tile", vec![Cond::all(&w1).step(), Cond::all(&[tile_lits(2), vec![w2]].concat()).step()]);
    let wn_glitch = Glitch {
        off: [wtile.clone(), wtile.clone()],
        off_attr: wtile + inserted.clone(),
        extra: [vec![([w1.clone(), vec![keep_low]].concat(), Source::Tile0(0))], vec![([w1, vec![keep_high]].concat(), Source::Tile0(1))]],
    };
    let bg = b.fetch("bg", &[bgon.clone(), nt(&winx), nt(&inserted)], &bsx, &bgrow, 3, &bg_glitch);
    let wn = b.fetch("wn", &[bgon.clone(), winx.clone(), nt(&inserted)], &wsx, &wrow, 6, &wn_glitch);
    let shown = b.pixel_at(&at, &bg, &wn, &winx, &bgon, &slots, &valid);

    // The palette lookup reads once per pixel, as the palette RAM does. Its query
    // names the pixel's palette line (`Shown::query`), a position-dependent query the
    // engine serves per position. CGB's colour is bytes 2c and 2c + 1 of the line for
    // colour c, the low or the high byte as FLG bit 2 asks; DMG's shade is bits 2c and
    // 2c + 1 of BGP, OBP0 or OBP1, bytes 0 to 2 of the DMG line.
    let level = b.c.level_of(&shown.line);
    let line = b.head("pal", level.max(b.c.level_of(&shown.colour)), shown.query(&cgb));
    let colour = |c: usize| -> Vec<Lin> { (0..2).map(|i| lit(&shown.colour[i], c >> i & 1)).collect() };
    let regs: [Vec<Lin>; 3] = [vec![nt(&shown.dmg_obj)], vec![shown.dmg_obj.clone(), nt(&shown.dmg_obp1)], vec![shown.dmg_obp1.clone()]];
    // DMG compatibility mode (FLG bit 3) draws as DMG, then looks the shade up in CGB
    // palette RAM: background palette 0 for BGP, object palette 0 or 1 for OBP0 or OBP1.
    // That line is known with the DMG one, so its head reads beside it.
    let compat = flg[3].clone();
    // Outside compatibility mode the query names PAL's first line at every position, so
    // the head reads one line and stays still: each varying bit is `x + compat − 1`,
    // whose sign is `x ∧ compat`.
    let gate = |x: &Lin| x.clone() + compat.clone() - one();
    let mut cq = line_base();
    cq.extend([gate(&shown.dmg_obp1), zero(), zero(), gate(&shown.dmg_obj), zero()]);
    for bit in 8..crate::circuit::ADDR_BITS {
        cq.push(k(((PAL >> bit) & 1) as i64));
    }
    let cline = b.head("cpal", level.max(b.c.level_of(&[shown.dmg_obj.clone(), shown.dmg_obp1.clone()])), Query::Wires(cq));
    // Compatibility mode reads the pixel's DMG shade (`Shown::dmg_shade`), ready at the
    // pixel's level, so each output bit takes 8 steps for it instead of 96.
    let dshade = shown.dmg_shade.clone();
    let open = nt(&flg[4]);
    let out: [Lin; 8] = std::array::from_fn(|j| {
        let mut steps: Vec<Step> = Vec::new();
        for c in 0..4 {
            steps.push(Cond::all(&[colour(c), vec![nt(&half), cgb.clone(), Lin::from(line[2 * c][j])]].concat()).step());
            if j + 8 < 15 {
                steps.push(Cond::all(&[colour(c), vec![half.clone(), cgb.clone(), Lin::from(line[2 * c + 1][j])]].concat()).step());
            }
            if j < 2 {
                for (r, reg) in regs.iter().enumerate() {
                    steps.push(Cond::all(&[colour(c), reg.clone(), vec![nt(&half), nt(&cgb), nt(&compat), Lin::from(line[r][2 * c + j])]].concat()).step());
                }
            }
        }
        for s in 0..4usize {
            let shade = vec![lit(&dshade[0], s & 1), lit(&dshade[1], s >> 1 & 1), compat.clone()];
            steps.push(Cond::all(&[shade.clone(), vec![nt(&half), Lin::from(cline[2 * s][j])]].concat()).step());
            if j + 8 < 15 {
                steps.push(Cond::all(&[shade, vec![half.clone(), Lin::from(cline[2 * s + 1][j])]].concat()).step());
            }
        }
        // Ordinary STOP's black: each step also needs FLG bit 4 clear.
        let steps = steps.into_iter().map(|s| Step { coef: s.coef, lin: s.lin + open.clone(), theta: s.theta + 1 }).collect();
        b.c.node(&format!("out.{j}"), steps)
    });
    for x in 0..WIDTH {
        b.c.output(&format!("PX{x}"), out.clone());
    }
    // The window counter counts the rows the line started (`Win::counted`); it arms at LY ==
    // WY. The window starts with LCDC.0 clear on DMG timing too, drawing white: its rows and
    // its 6 dots count (`machine::fifo`).
    let wln_n = b.c.sum_bits("WLN'", 8, |n| value(&wln, n) + win.counted.clone());
    let cgbhw = flg[1].clone() + flg[3].clone();
    let (length, _) = mode3_length(&mut b.c, &xsel, &valid, &scx, &wx, &win, &cgbhw);
    b.c.output(WINDOW[0], wln_n.try_into().unwrap());
    b.c.output(WINDOW[1], window_out(&win));
    b.c.output(LENGTH, length);
    record_slots(&mut b.c);
    b.c.schedule(crate::llama::UNIT_QUANTUM);
    b.c
}

/// The line's length as the PPU publishes it (`fifo::length`: the dots to mode 0 less
/// `lcd::LENGTH_BASE`, a DMG's fine scroll rounded up to 4, `cgb` CGB hardware), and the dots (the FIFO's, `machine::fifo`, on a line
/// with no write; `fifo::length_rule::closed_form`): 172 dots, plus SCX's fine scroll, 6 when
/// the window starts (one more at WX 0 under a fine scroll), and 6 per object fetched (X
/// below 168), the first object in each tile (the least X, ties to OAM order) also waiting
/// `5 − min(5, fine position)`. An object at X 0 is at 0 of tile 0 whatever SCX (Pan Docs,
/// "OBJ penalty algorithm"). Otherwise its tile and fine position are those of `X + (SCX & 7)`
/// in the background's grid; of `X − WX − 1` in the window's, at or past the window's start;
/// past an inserted pixel at P, of `X − 9 − P` in the grid a dot later, and on a window line
/// first in one tile 9 positions wide, at `X − 8 − P`. `xs` are the scan's slots' X in OAM
/// order, `valid` whether each holds an object.
fn mode3_length(c: &mut Circuit, xs: &[W], valid: &[Lin], scx: &W, wx: &W, win: &Win, cgb: &Lin) -> ([Lin; 8], Lin) {
    let n = xs.len();
    let f = value(scx, 3);
    let fetched: W = (0..n).map(|s| c.ge(&format!("m3 fetched{s}"), valid[s].clone() * 168 + k(167) - value(&xs[s], 8), 168)).collect();
    // X + (SCX & 7): the tile is bits 3-7, the fine position bits 0-2. A slot whose sum
    // passes 255 is not fetched.
    let bg: Vec<W> = (0..n).map(|s| c.sum_bits(&format!("m3 at{s}"), 8, |j| value(&xs[s], j) + value(scx, j.min(3)))).collect();
    // X − WX − 1 modulo 256, the window's grid, and X + f + 7, the grid past an inserted pixel:
    // `X − 9 − P` is `X − WX − 2` at P = WX − 7 and `X + f − 9` at P = −f, and an insertion at
    // WX − 7 has `(WX & 7) + f == 7`, so both are `X + f + 7` less a multiple of 8, the same
    // per line: the same fine positions, the same tiles. On a window line the 9 positions
    // from P count at `d = X − WX − 1`, whose low bits are those of `X + f` there.
    let win1: Vec<W> = (0..n).map(|s| c.sum_bits(&format!("m3 atw{s}"), 8, |j| value(&xs[s], j) + k((1 << j) - 1) - value(wx, j))).collect();
    let junk: Vec<W> = (0..n).map(|s| c.sum_bits(&format!("m3 atj{s}"), 8, |j| value(&xs[s], j) + value(scx, j.min(3)) + k(7 & ((1 << j) - 1)))).collect();
    let x0: W = (0..n).map(|s| c.and(&format!("m3 x0.{s}"), &(0..8).map(|i| nt(&xs[s][i])).collect::<Vec<_>>())).collect();
    // Per slot the grid it counts in, the cases exclusive.
    let d = |s: usize| value(&xs[s], 8) - value(wx, 8) - k(1);
    let at: Vec<W> = (0..n)
        .map(|s| {
            let nx0 = nt(&x0[s]);
            let on_win = c.node(&format!("m3 onwin{s}"), vec![Cond::all(&[win.std.clone(), nx0.clone()]).ge(d(s), 0, 255)]);
            let row = Cond::all(&[win.in_row.clone(), nx0.clone()]);
            let block = c.node(&format!("m3 block{s}"), vec![row.ge(d(s), 0, 255), { let mut st = row.ge(d(s), 8, 255); st.coef = -1; st }]);
            let block8 = c.node(&format!("m3 block8.{s}"), row.eq(d(s), 8, 255).to_vec());
            let past = c.node(
                &format!("m3 past{s}"),
                vec![Cond::all(&[win.i278.clone(), nx0.clone()]).ge(d(s), 0, 255), row.ge(d(s), 9, 255), Cond::all(&[win.i247.clone(), nx0.clone()]).ge(value(&xs[s], 8) + f.clone(), 8, 262)],
            );
            let on_bg = nx0 - on_win.clone() - block.clone() - block8.clone() - past.clone();
            let mut bits: W = (0..8)
                .map(|j| {
                    let mut steps = vec![
                        Cond::all(&[on_bg.clone(), bg[s][j].clone()]).step(),
                        Cond::all(&[on_win.clone(), win1[s][j].clone()]).step(),
                        Cond::all(&[past.clone(), junk[s][j].clone()]).step(),
                    ];
                    if j < 3 {
                        steps.push(Cond::all(&[block.clone(), bg[s][j].clone()]).step());
                        steps.push(Cond::all(&[block8.clone()]).step());
                    }
                    c.node(&format!("m3 pos{s}.{j}"), steps)
                })
                .collect();
            bits.push(on_win + past);
            bits.push(block + block8);
            bits
        })
        .collect();
    let same = |c: &mut Circuit, p: usize, q: usize| c.node(&format!("m3 tile{p}={q}"), eq2(value(&at[p][3..], 7) - value(&at[q][3..], 7), 0));
    let mut tile_eq = vec![vec![zero(); n]; n];
    let mut le = vec![vec![zero(); n]; n];
    for p in 0..n {
        for q in p + 1..n {
            tile_eq[p][q] = same(c, p, q);
            le[p][q] = c.ge(&format!("m3 x{p}<={q}"), value(&xs[q], 8) - value(&xs[p], 8), 0);
        }
    }
    // Slot t comes before slot s in the fetch order: a lesser X, or an equal one earlier in OAM.
    let precedes = |t: usize, s: usize| if t < s { le[t][s].clone() } else { nt(&le[s][t]) };
    let ahead: Vec<W> = (0..n)
        .map(|s| (0..n).filter(|&t| t != s).map(|t| c.ge(&format!("m3 ahead{t}.{s}"), fetched[t].clone() + tile_eq[t.min(s)][t.max(s)].clone() + precedes(t, s), 3)).collect())
        .collect();
    // The first object of its tile, with fine position v below 5: it waits 5 − v dots.
    let mut extra = f + win.start.clone() * 6 + win.wx0start.clone();
    for s in 0..n {
        extra = extra + fetched[s].clone() * 6;
        let others = sum(ahead[s].iter().cloned());
        for v in 0..5 {
            let lits = [vec![fetched[s].clone()], (0..3).map(|i| lit(&at[s][i], v >> i & 1)).collect()].concat();
            let first = c.node(&format!("m3 first{s}@{v}"), vec![Step { coef: 1, lin: sum(lits.into_iter()) * 10 - others.clone(), theta: 40 }]);
            extra = extra + first * (5 - v as i64);
        }
    }
    let dots = extra + 172;
    // On a DMG SCX's fine scroll ends mode 3 at the next M-cycle's boundary
    // (`fifo::mode0_dots`): the length counts it rounded up to 4, (−fine) mod 4 more, fine
    // bit 0 plus twice bit 0 xor bit 1. A CGB's is the dots. The dots stay exact for the judge.
    let x01 = c.eq("m3 fine xor", scx[0].clone() + scx[1].clone(), 1);
    let f0 = c.and("m3 fine.0 dmg", &[scx[0].clone(), nt(cgb)]);
    let x01 = c.and("m3 fine xor dmg", &[x01, nt(cgb)]);
    let published = dots.clone() + f0 + x01 * 2 - crate::machine::lcd::LENGTH_BASE as i64;
    (std::array::from_fn(|i| c.bit_of(&format!("ML.{i}"), published.clone(), i as u32)), dots)
}

/// The input bytes of a line in `INPUTS` order; `high` asks for the colour's high byte.
pub fn inputs(ly: u8, window: Window, mode: Mode, high: bool, blocked: bool) -> [u8; 3] {
    [ly, window.rows, window.armed as u8 | if window.carried { CARRIED_IN } else { 0 } | (mode.cgb_features() as u8) << 1 | (high as u8) << 2 | ((mode == Mode::Compat) as u8) << 3 | (blocked as u8) << 4]
}

/// The length circuit's input slots: LY and FLG (bit 0 the window armed, bit 1 CGB mode,
/// bit 3 DMG compatibility mode, bit 5 the window carried in, bit 6 a write to judge), as the
/// scanline circuit takes them; then the write: the machine's 2-dot halves since the line's
/// mode 3 began, its display index and its value.
pub const LENGTH_INPUTS: [&str; 5] = ["MLY", "MFLG", "MWH", "MWI", "MWV"];
/// Its outputs: the line's mode-3 length and the window's arming and carry after it, and
/// the judge of the write (`JUDGE_MARK`, `JUDGE_TAKE`) and its time in half-dots
/// (`fifo::Time`, two bytes), which the stepper takes.
pub const LENGTH_OUTPUTS: [&str; 5] = ["MML", "MWAR'", "MJ", "MWT0", "MWT1"];
/// MFLG: a write to judge.
pub const LENGTH_WRITE: u8 = 0x40;
/// MJ: the write can change the line (`fifo::judge`), so the stepper draws it.
pub const JUDGE_MARK: u8 = 1;
/// MJ: the write is to a display register, which the stepper takes in its queue.
pub const JUDGE_TAKE: u8 = 2;

/// The line's mode-3 length on its own (`docs/ppu-in-weights.md`): the scanline circuit's
/// OAM scan, its window test and `mode3_length`, over the same OAM image and display
/// registers, and no pixel. It answers a line's length as the line is handed over, so
/// the control logic never waits for the renderer; the renderer's own `ML` is held to it
/// line by line. A pass with a write also judges it (`judge`).
pub fn build_length() -> Circuit {
    let mut c = Circuit::default();
    let ly = w(c.input(LENGTH_INPUTS[0]));
    let flg = w(c.input(LENGTH_INPUTS[1]));
    let write = [2, 3, 4].map(|i| w(c.input(LENGTH_INPUTS[i])));
    let registers = regs(&mut c);
    let [bgp, obp0, obp1, lcdc, _scy, scx, wy, wx] = registers.clone();
    let cgb = flg[1].clone();
    let mut b = B { c, lcdc: lcdc.clone(), cgb: cgb.clone(), pals: [bgp, obp0, obp1], compat: flg[3].clone() };
    let win = window_cases(&mut b, &ly, &flg, &scx, &wy, &wx);
    let Scan { valid, xsel, .. } = b.scan(&ly, None, false);
    let cgbhw = flg[1].clone() + flg[3].clone();
    let (length, dots) = mode3_length(&mut b.c, &xsel, &valid, &scx, &wx, &win, &cgbhw);
    b.c.output(LENGTH_OUTPUTS[0], length);
    b.c.output(LENGTH_OUTPUTS[1], window_out(&win));
    let [mj, t0, t1] = judge(&mut b.c, &flg, &write, &registers, &dots);
    b.c.output(LENGTH_OUTPUTS[2], mj);
    b.c.output(LENGTH_OUTPUTS[3], t0);
    b.c.output(LENGTH_OUTPUTS[4], t1);
    record_slots(&mut b.c);
    b.c.schedule(crate::llama::UNIT_QUANTUM);
    b.c
}

/// `fifo::judge` on the pass's write (`MWH`, `MWI`, `MWV`), with the write's time from its
/// halves, `4·H + 8` half-dots (`ppu::WRITE_OFFSET`), and the quiet line's `dots`. The time
/// leaves as `MWT0`/`MWT1`, so the stepper's queue carries the model's time, not the host's.
fn judge(c: &mut Circuit, flg: &W, [h, index, v]: &[W; 3], registers: &[W; 8], dots: &Lin) -> [[Lin; 8]; 3] {
    const { assert!(crate::machine::ppu::WRITE_OFFSET == 8 && fifo::REACH <= 8) };
    let sel: Vec<Lin> = (0..8).map(|r| c.eq(&format!("mj sel{r}"), value(index, 8), PAL_DMG as i64 + r as i64)).collect();
    let same: Vec<Lin> = (0..8).map(|r| c.eq(&format!("mj same{r}"), value(v, 8) - value(&registers[r], 8), 0)).collect();
    // The DMG's WX comparator notes any write to WX, its value or not (`fifo::act`).
    let dmg = one() - flg[1].clone() - flg[3].clone();
    let mut steps: Vec<Step> = (0..7).map(|r| Step { coef: 1, lin: sel[r].clone() + same[r].clone(), theta: 2 }).collect();
    steps.push(Step { coef: 1, lin: sel[7].clone() + same[7].clone() + nt(&dmg), theta: 3 });
    let same = c.node("mj same", steps);
    // WY, or BGP, OBP0 and OBP1 in CGB mode: read by no pixel of this line.
    let pal = c.node("mj cgb palette", (0..3).map(|r| Step { coef: 1, lin: flg[1].clone() + sel[r].clone(), theta: 2 }).collect());
    let unread = sel[6].clone() + pal;
    // 4·H + 8 ≥ 2·dots + REACH. Both sides are even but REACH, so it is 2·H + 4 ≥ dots +
    // ⌈REACH / 2⌉, halved to keep H's weights within f16's.
    let late = c.ge("mj late", value(h, 8) * 2 - dots.clone() + k(4 - (fifo::REACH as i64 + 1) / 2), 0);
    let register = sum(sel.iter().cloned());
    let mark = c.ge("mj mark", flg[6].clone() + register.clone() - late - same - unread, 2);
    let take = c.and("mj take", &[flg[6].clone(), register]);
    let h2 = c.sum_bits("mj h+2", 9, |j| value(h, j.min(8)) + k(2 & ((1 << j) - 1)));
    let time: Vec<Lin> = (0..16).map(|b| if (2..11).contains(&b) { h2[b - 2].clone() } else { zero() }).collect();
    [std::array::from_fn(|i| [mark.clone(), take.clone()].get(i).cloned().unwrap_or_else(zero)), std::array::from_fn(|i| time[i].clone()), std::array::from_fn(|i| time[8 + i].clone())]
}

/// The length circuit's input bytes, `LENGTH_INPUTS` order: a line's pass, or with `write`
/// (halves since mode 3 began, display index, value) a pass that judges it too.
pub fn length_inputs(ly: u8, window: Window, mode: Mode, write: Option<(u8, u8, u8)>) -> [u8; 5] {
    let flg = window.armed as u8 | if window.carried { CARRIED_IN } else { 0 } | (mode.cgb_features() as u8) << 1 | ((mode == Mode::Compat) as u8) << 3;
    let (h, i, v) = write.unwrap_or((0, 0, 0));
    [ly, flg | if write.is_some() { LENGTH_WRITE } else { 0 }, h, i, v]
}

/// The write-record slots the memory compiler expects, which a PPU pass leaves zero.
pub(crate) fn record_slots(c: &mut Circuit) {
    for wr in 0..2u8 {
        for (byte, n) in ["a", "b", "c"].iter().enumerate() {
            c.output_kind(&format!("WP{wr}{n}"), std::array::from_fn(|_| zero()), OutKind::Address { write: wr, byte: byte as u8 });
        }
        for byte in 0..LINE as u8 {
            c.output_kind(&format!("WL{wr}{byte}"), std::array::from_fn(|_| zero()), OutKind::Line { write: wr, byte });
        }
    }
}
