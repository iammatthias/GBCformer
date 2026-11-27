//! Which layer computes each node and holds each memory head.
//!
//! A stock runtime's file is uniform: every layer stores and computes the widest layer's
//! MLP and the widest layer's heads. In a checkpoint of several parts a layer's units are
//! the sum over the parts placed there, and so are its heads, so the uniform cost is the
//! layer count times the widest sum. A circuit fixes only an order: a node needs its
//! inputs from an earlier layer, a head its query, and the residual keeps every wire for
//! as long as it takes. [`Graph`] holds that order for one circuit, and [`Graph::pack`]
//! places its free items in the room the others leave in each layer, so short parts fill
//! the layers a deep part needs anyway. No value changes: only the layer that computes
//! it (`docs/perf-log.md`, "Re-levelling").

use crate::circuit::{Circuit, Query, LINE_BITS};

/// One circuit's placement problem. An item is a node, or the memory heads that drive
/// one bus and so share a layer (a head alone when it drives none).
///
/// A node of level x sits in layer x − 1's MLP and reads wires of level below x. A head
/// at layer x reads its query from wires of level at most x, and writes wires of level
/// x, which a node reads from level x + 1 and another head's query from layer x + 1.
/// Outputs are read at the circuit's depth, its deepest level.
pub struct Graph {
    /// Items below this are the circuit's nodes, in order; the rest are head groups.
    nodes: usize,
    /// Per head group, its heads.
    groups: Vec<Vec<usize>>,
    /// Per item, each item it reads and how many levels later it must sit (0 for a head
    /// reading a node, 1 otherwise).
    reads: Vec<Vec<(u32, u32)>>,
    /// Per item, each item that reads it, with the same gap.
    readers: Vec<Vec<(u32, u32)>>,
    /// Per item, its MLP units (0 for heads).
    units: Vec<usize>,
    /// Per item, its level as the circuit has it.
    original: Vec<u32>,
    /// The items, each after everything it reads.
    order: Vec<u32>,
    /// Per item, the wires it reads and the wires it writes, those that hold a residual
    /// channel (not a position wire, not a line bit nothing reads).
    ins: Vec<Vec<u32>>,
    outs: Vec<Vec<u32>>,
    /// Per wire, whether an output reads it.
    routed: Vec<bool>,
}

/// What each layer has room for: MLP units per MLP layer (in `Request::cost`'s scale)
/// and heads per layer.
///
/// The caps are soft, and the room says so by going below zero. A pinned item sits
/// where it is, and a free item with no later level left sits at its earliest, whatever
/// room is left there; `examples/room -- --scan` relies on that to try caps below what
/// fits. A unit's cost is also an estimate (its part's clearing units charged in
/// proportion), so what a placement compiles to is `Model::census`'s to say. Nothing is
/// hidden: [`Room::over`] is how far the fullest layer ended past its cap, and
/// `one_model::circuit` asserts the canonical placement ends within both.
#[derive(Clone, Debug)]
pub struct Room {
    pub units: Vec<i64>,
    pub heads: Vec<i64>,
}

/// How far a placement ended past its caps: the most scaled units a layer holds beyond
/// its room, and the most heads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Over {
    pub units: usize,
    pub heads: usize,
}

impl Room {
    pub fn over(&self) -> Over {
        let most = |left: &[i64]| left.iter().map(|&l| (-l).max(0) as usize).max().unwrap_or(0);
        Over { units: most(&self.units), heads: most(&self.heads) }
    }
}

/// What `Graph::pack` is asked: a circuit of `depth` levels whose items sit in levels
/// 1..=`depth`.
pub struct Request<'a> {
    pub depth: u32,
    /// Per item, the level it is held at, or 0 for a free item.
    pub pins: &'a [u32],
    /// Per free item, its last level: its part's depth.
    pub last: &'a [u32],
    /// Per item, what it takes of a layer's unit room.
    pub cost: &'a [usize],
    /// Per wire, its part.
    pub part: &'a [u32],
    /// The wires a part should keep live at most.
    pub live: usize,
}

impl Graph {
    pub fn of(c: &Circuit) -> Graph {
        let nodes = c.nodes.len();
        const NONE: u32 = u32::MAX;
        let mut producer = vec![NONE; c.levels.len()];
        for (i, nd) in c.nodes.iter().enumerate() {
            producer[nd.out.0 as usize] = i as u32;
        }
        // Heads that share an output wire drive one bus: one group, one layer.
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (h, head) in c.heads.iter().enumerate() {
            let mut joined: Vec<usize> = head.out.iter().filter_map(|w| (producer[w.0 as usize] != NONE).then(|| (producer[w.0 as usize] as usize).checked_sub(nodes).unwrap_or_else(|| panic!("{}: a head writes a node's wire", head.name)))).collect();
            joined.sort_unstable();
            joined.dedup();
            let g = match joined.first() {
                Some(&g) => g,
                None => {
                    groups.push(Vec::new());
                    groups.len() - 1
                }
            };
            for &other in joined.iter().skip(1) {
                let moved = std::mem::take(&mut groups[other]);
                groups[g].extend(moved);
                for p in producer.iter_mut().filter(|p| **p == (nodes + other) as u32) {
                    *p = (nodes + g) as u32;
                }
            }
            groups[g].push(h);
            for w in &head.out {
                producer[w.0 as usize] = (nodes + g) as u32;
            }
        }
        // Merged groups leave empty ones behind; renumber the rest.
        let mut renumber = vec![usize::MAX; groups.len()];
        let mut kept: Vec<Vec<usize>> = Vec::new();
        for (g, heads) in groups.into_iter().enumerate() {
            if !heads.is_empty() {
                renumber[g] = kept.len();
                kept.push(heads);
            }
        }
        for p in producer.iter_mut().filter(|p| **p != NONE && **p >= nodes as u32) {
            *p = (nodes + renumber[*p as usize - nodes]) as u32;
        }
        let groups = kept;
        let n = nodes + groups.len();
        let unread = c.unread();
        let channel = |w: usize| !c.is_position(w) && !unread[w];
        let mut reads: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
        let mut ins: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut outs: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (i, nd) in c.nodes.iter().enumerate() {
            outs[i].push(nd.out.0);
            for w in nd.steps.iter().flat_map(|s| s.lin.terms.keys()) {
                ins[i].extend(channel(w.0 as usize).then_some(w.0));
                match producer[w.0 as usize] {
                    NONE => assert_eq!(c.level(*w), 0, "{}: reads wire {}, which nothing writes", nd.name, c.names[w.0 as usize]),
                    p => reads[i].push((p, 1)),
                }
            }
        }
        for (g, heads) in groups.iter().enumerate() {
            for &h in heads {
                outs[nodes + g].extend(c.heads[h].out.iter().filter(|w| channel(w.0 as usize)).map(|w| w.0));
                if let Query::Wires(q) = &c.heads[h].query {
                    for w in q[LINE_BITS..].iter().flat_map(|l| l.terms.keys()) {
                        ins[nodes + g].extend(channel(w.0 as usize).then_some(w.0));
                        match producer[w.0 as usize] {
                            NONE => assert_eq!(c.level(*w), 0, "{}: its query reads wire {}, which nothing writes", c.heads[h].name, c.names[w.0 as usize]),
                            p => reads[nodes + g].push((p, u32::from(p as usize >= nodes))),
                        }
                    }
                }
            }
        }
        let mut readers: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
        for (i, r) in reads.iter_mut().enumerate() {
            r.sort_unstable();
            r.dedup();
            for &(j, gap) in r.iter() {
                readers[j as usize].push((i as u32, gap));
            }
        }
        for v in ins.iter_mut().chain(&mut outs) {
            v.sort_unstable();
            v.dedup();
        }
        let mut routed = vec![false; c.levels.len()];
        for w in c.output_lins().flat_map(|l| l.terms.keys()) {
            routed[w.0 as usize] = true;
        }
        let units = c.nodes.iter().map(|nd| c.units(nd)).chain(groups.iter().map(|_| 0)).collect();
        let original: Vec<u32> = c.nodes.iter().map(|nd| c.level(nd.out)).chain(groups.iter().map(|g| c.heads[g[0]].layer)).collect();
        for (g, heads) in groups.iter().enumerate() {
            assert!(heads.iter().all(|&h| c.heads[h].layer == original[nodes + g]), "the heads of one bus sit at one layer");
        }
        // Each item after everything it reads.
        let mut waiting: Vec<usize> = reads.iter().map(Vec::len).collect();
        let mut order: Vec<u32> = (0..n as u32).filter(|&i| waiting[i as usize] == 0).collect();
        let mut at = 0;
        while at < order.len() {
            for &(r, _) in &readers[order[at] as usize] {
                waiting[r as usize] -= 1;
                if waiting[r as usize] == 0 {
                    order.push(r);
                }
            }
            at += 1;
        }
        assert_eq!(order.len(), n, "the circuit's reads form a cycle");
        Graph { nodes, groups, reads, readers, units, original, order, ins, outs, routed }
    }

    pub fn len(&self) -> usize {
        self.units.len()
    }

    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    pub fn is_node(&self, i: usize) -> bool {
        i < self.nodes
    }

    /// Item `i`'s MLP units and heads.
    pub fn weight(&self, i: usize) -> (usize, usize) {
        (self.units[i], if i < self.nodes { 0 } else { self.groups[i - self.nodes].len() })
    }

    /// Each item's level as the circuit has it.
    pub fn original(&self) -> &[u32] {
        &self.original
    }

    /// Each item's earliest level. `pins` holds an item at a level (0 for a free item).
    pub fn earliest(&self, pins: &[u32]) -> Vec<u32> {
        let mut x = vec![1u32; self.len()];
        for &i in &self.order {
            let i = i as usize;
            let floor = self.reads[i].iter().map(|&(j, gap)| x[j as usize] + gap).max().unwrap_or(1).max(1);
            x[i] = if pins[i] != 0 { pins[i] } else { floor };
            assert!(x[i] >= floor, "a pinned item sits before what it reads");
        }
        x
    }

    /// Each item's latest level in a circuit of `depth` levels (below 1 when `depth` is
    /// too few).
    pub fn latest(&self, depth: u32, pins: &[u32]) -> Vec<i64> {
        let mut x = vec![depth as i64; self.len()];
        for &i in self.order.iter().rev() {
            let i = i as usize;
            let cap = self.readers[i].iter().map(|&(r, gap)| x[r as usize] - gap as i64).min().unwrap_or(depth as i64);
            x[i] = if pins[i] != 0 { pins[i] as i64 } else { cap };
        }
        x
    }

    /// Places every item in a level of 1..=`depth`, taking each layer's room as it goes,
    /// or returns None when a part's depth is too few for it.
    ///
    /// It fills the levels from the last down, so a part ends at its depth and leaves
    /// the first layers to others. A level takes the heads whose readers are all placed,
    /// then nodes whose readers are: first those at their earliest level, whatever the
    /// room, then as long as they fit, one at a time, a node within `URGENT` levels of its
    /// earliest before any other, and among those the node that adds the fewest live
    /// wires below the level (the wires it reads that nothing placed reads yet, less the
    /// wire it writes), the one with the higher earliest level among equals; a node that
    /// would take its part past `live` waits for a lower level. Fewer live wires is
    /// fewer residual channels.
    pub fn pack(&self, ask: &Request, room: &mut Room) -> Option<Vec<u32>> {
        let &Request { depth, pins, last, cost, part, live } = ask;
        let n = self.len();
        let early = self.earliest(pins);
        if (0..n).any(|i| pins[i] == 0 && early[i] > last[i]) {
            return None;
        }
        let mut level = pins.to_vec();
        for i in (0..n).filter(|&i| pins[i] != 0) {
            room.units[pins[i] as usize - 1] -= cost[i] as i64;
            room.heads[pins[i] as usize] -= self.weight(i).1 as i64;
        }
        let mut left: Vec<usize> = (0..n).filter(|&i| pins[i] == 0).collect();
        // The wires something placed reads (an output, from the start) and nothing
        // placed writes: live below the level being filled. `lives` counts them by part.
        let mut open = self.routed.clone();
        let mut lives = vec![0usize; part.iter().map(|&p| p as usize + 1).max().unwrap_or(1)];
        for w in (0..open.len()).filter(|&w| open[w]) {
            lives[part[w] as usize] += 1;
        }
        let place = |i: usize, x: u32, level: &mut [u32], open: &mut [bool], lives: &mut [usize]| {
            level[i] = x;
            for &w in &self.outs[i] {
                lives[part[w as usize] as usize] -= open[w as usize] as usize;
                open[w as usize] = false;
            }
            for &w in &self.ins[i] {
                lives[part[w as usize] as usize] += !open[w as usize] as usize;
                open[w as usize] = true;
            }
        };
        for x in (1..=depth).rev() {
            // A head at this layer may read a node of this level, so heads go first.
            for heads in [true, false] {
                let mut ready: Vec<usize> = left.iter().copied().filter(|&i| (i >= self.nodes) == heads && x <= last[i] && self.readers[i].iter().all(|&(r, gap)| level[r as usize] >= x + gap)).collect();
                ready.sort_by_key(|&i| (u32::MAX - early[i], i));
                if heads {
                    for i in ready {
                        let (have, need) = (&mut room.heads[x as usize], self.weight(i).1 as i64);
                        if *have >= need || early[i] == x {
                            *have -= need;
                            place(i, x, &mut level, &mut open, &mut lives);
                        }
                    }
                    continue;
                }
                let have = &mut room.units[x as usize - 1];
                for &i in ready.iter().filter(|&&i| early[i] == x) {
                    *have -= cost[i] as i64;
                    place(i, x, &mut level, &mut open, &mut lives);
                }
                ready.retain(|&i| level[i] == 0);
                loop {
                    let added = |i: usize| self.ins[i].iter().filter(|&&w| !open[w as usize]).count() as i64 - self.outs[i].iter().filter(|&&w| open[w as usize]).count() as i64;
                    let fits = |i: usize| cost[i] as i64 <= (*have).max(0) && (added(i) <= 0 || lives[part[self.outs[i][0] as usize] as usize].saturating_add(added(i) as usize) <= live);
                    let Some(&i) = ready.iter().filter(|&&i| fits(i)).min_by_key(|&&i| ((x - early[i]).min(URGENT), added(i), u32::MAX - early[i], i)) else { break };
                    *have -= cost[i] as i64;
                    place(i, x, &mut level, &mut open, &mut lives);
                    ready.retain(|&j| j != i);
                }
            }
            left.retain(|&i| level[i] == 0);
        }
        left.is_empty().then_some(level)
    }

    /// The part of a circuit of several parts item `i` belongs to.
    pub fn part(&self, c: &Circuit, i: usize) -> usize {
        c.part_of(if i < self.nodes { c.nodes[i].out.0 } else { c.heads[self.groups[i - self.nodes][0]].out[0].0 } as usize)
    }

    /// Gives the circuit the levels `pack` returned: each node's wire, each head's layer
    /// and the wires it writes.
    pub fn apply(&self, c: &mut Circuit, levels: &[u32]) {
        for (i, nd) in c.nodes.iter().enumerate() {
            c.levels[nd.out.0 as usize] = levels[i];
        }
        for (g, heads) in self.groups.iter().enumerate() {
            for &h in heads {
                c.heads[h].layer = levels[self.nodes + g];
                for w in &c.heads[h].out {
                    c.levels[w.0 as usize] = levels[self.nodes + g];
                }
            }
        }
        check(c);
    }
}

/// A node this many levels above its earliest, or fewer, goes before one with more room
/// below it (`Graph::pack`), so a part with little depth to spare keeps the layers it can
/// use.
const URGENT: u32 = 3;

/// The scale of `Graph::pack`'s unit costs and room: this many to one MLP unit.
const SCALE: usize = 1024;

/// Re-levels the parts `free` of a circuit of several parts (`Circuit::concat`), each to
/// the depth given with it, in the room the other parts leave of `cap` MLP units and
/// `heads` heads a layer; the other parts stay as they are, and so does the circuit's
/// depth. `spoken` is what each layer holds besides nodes and clearing units (routing
/// units), and `clears` each part's clearing units, which a layer is charged in
/// proportion to the part's node units there; `live` is the wires a free part should
/// keep live at most. A layer may end over the cap (`Room`): the result says by how
/// much, in MLP units and heads, and `Model::census` has the shape. None, with the
/// circuit unchanged, when a depth is too few for its part.
pub fn level(c: &mut Circuit, free: &[(usize, u32)], spoken: &[usize], clears: &[usize], cap: usize, heads: usize, live: usize) -> Option<Over> {
    let g = Graph::of(c);
    let depth = c.depth();
    let part: Vec<usize> = (0..g.len()).map(|i| g.part(c, i)).collect();
    let mut nodes = vec![0usize; clears.len()];
    for i in 0..g.len() {
        nodes[part[i]] += g.units[i];
    }
    let cost: Vec<usize> = (0..g.len()).map(|i| (g.units[i] * SCALE * (nodes[part[i]] + clears[part[i]])).div_ceil(nodes[part[i]].max(1))).collect();
    let ends = |p: usize| free.iter().find(|f| f.0 == p).map(|f| f.1);
    let pins: Vec<u32> = (0..g.len()).map(|i| if ends(part[i]).is_some() { 0 } else { g.original[i] }).collect();
    let last: Vec<u32> = (0..g.len()).map(|i| ends(part[i]).unwrap_or(depth)).collect();
    assert!(last.iter().all(|&l| l <= depth), "a part's depth is past the circuit's");
    let mut room = Room { units: (0..depth as usize).map(|l| SCALE as i64 * (cap as i64 - spoken[l] as i64)).collect(), heads: vec![heads as i64; depth as usize + 1] };
    let wire_part: Vec<u32> = (0..c.levels.len()).map(|w| c.part_of(w) as u32).collect();
    let levels = g.pack(&Request { depth, pins: &pins, last: &last, cost: &cost, part: &wire_part, live }, &mut room)?;
    g.apply(c, &levels);
    let over = room.over();
    Some(Over { units: over.units.div_ceil(SCALE), ..over })
}

/// Asserts the order every placement must keep: a node above every wire it reads, a
/// head's query at or below its layer (below it for a wire another head writes), and a
/// head's wires at its layer.
pub fn check(c: &Circuit) {
    let mut from_head = vec![false; c.levels.len()];
    for w in c.heads.iter().flat_map(|h| &h.out) {
        from_head[w.0 as usize] = true;
    }
    for nd in &c.nodes {
        let level = c.level(nd.out);
        for w in nd.steps.iter().flat_map(|s| s.lin.terms.keys()) {
            assert!(c.level(*w) < level, "{} (level {level}) reads {} (level {})", nd.name, c.names[w.0 as usize], c.level(*w));
        }
    }
    for h in &c.heads {
        assert!(h.layer >= 1, "{}: a memory head at layer 0", h.name);
        assert!(h.out.iter().all(|&w| c.level(w) == h.layer), "{}: a wire it writes is not of its layer {}", h.name, h.layer);
        if let Query::Wires(q) = &h.query {
            for w in q[LINE_BITS..].iter().flat_map(|l| l.terms.keys()) {
                assert!(c.level(*w) + u32::from(from_head[w.0 as usize]) <= h.layer, "{} (layer {}) reads {} (level {})", h.name, h.layer, c.names[w.0 as usize], c.level(*w));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{byte, zero, Lin, OutKind};
    use std::collections::BTreeMap;

    /// A chain `depth` levels deep with one unit a level.
    fn chain(depth: usize) -> Circuit {
        let mut c = Circuit::default();
        let x = byte(c.input("X"));
        let mut out: [Lin; 8] = std::array::from_fn(|_| zero());
        out[0] = (1..=depth).fold(x[0].clone(), |chain, k| c.and(&format!("c{k}"), &[chain, x[k].clone()]));
        c.output("Y", out);
        c
    }

    /// A chain six levels deep beside a part two levels deep that holds six units in its
    /// second.
    fn two_parts() -> Circuit {
        let mut wide = Circuit::default();
        let x = byte(wide.input("X"));
        let a = wide.and("a", &[x[0].clone(), x[1].clone()]);
        let mut out: [Lin; 8] = std::array::from_fn(|_| zero());
        for k in 2..8 {
            out[k] = wide.and(&format!("n{k}"), &[a.clone(), x[k].clone()]);
        }
        wide.output("Y", out);
        side_by_side(&[("deep", &chain(6)), ("wide", &wide)])
    }

    /// A part whose two heads drive one bus: `m0` could sit a layer before `m1`, whose
    /// query reads a deeper node.
    fn bus_part() -> Circuit {
        let mut m = Circuit::default();
        let x = byte(m.input("X"));
        let a = m.and("a", &[x[0].clone(), x[1].clone()]);
        let b = m.and("b", &[a.clone(), x[2].clone()]);
        let bus = m.bus("bus", 2);
        m.head_onto("m0", 2, line_of(&a), &[(0, bus)]);
        m.head_onto("m1", 2, line_of(&b), &[(0, bus)]);
        let mut out: [Lin; 8] = std::array::from_fn(|_| zero());
        out[0] = m.and("r", &[Lin::from(bus[0]), x[3].clone()]);
        m.output("Y", out);
        m
    }

    /// A part with one head, whose query reads `a` and whose line `r` reads.
    fn head_part() -> Circuit {
        let mut m = Circuit::default();
        let x = byte(m.input("X"));
        let a = m.and("a", &[x[0].clone(), x[1].clone()]);
        let line = m.head("m", 1, line_of(&a));
        let mut out: [Lin; 8] = std::array::from_fn(|_| zero());
        out[0] = m.and("r", &[Lin::from(line[0][0]), x[2].clone()]);
        m.output("Y", out);
        m
    }

    /// The query for the line `bit` names: line 1 or line 0.
    fn line_of(bit: &Lin) -> Query {
        let mut address: Vec<Lin> = (0..crate::circuit::ADDR_BITS).map(|_| zero()).collect();
        address[LINE_BITS] = bit.clone();
        Query::Wires(address)
    }

    fn side_by_side(parts: &[(&str, &Circuit)]) -> Circuit {
        Circuit::concat(parts, |_, kind| kind == OutKind::Shared).0
    }

    fn node(c: &Circuit, name: &str) -> u32 {
        c.level(c.nodes.iter().find(|n| n.name == name).unwrap_or_else(|| panic!("no node {name}")).out)
    }

    const FITS: Option<Over> = Some(Over { units: 0, heads: 0 });

    #[test]
    fn a_free_part_fills_the_layers_the_other_leaves() {
        let mut c = two_parts();
        assert_eq!(c.units_per_level(), vec![2, 7, 1, 1, 1, 1]);
        assert_eq!(level(&mut c, &[(1, 6)], &[0; 7], &[0, 0], 3, 0, usize::MAX), FITS);
        assert_eq!(c.units_per_level(), vec![1, 1, 2, 3, 3, 3]);
        assert_eq!((c.part_depth(0), c.part_depth(1)), (6, 6));
        // The pinned chain stays; the free part's `a` sits below every node that reads it.
        assert_eq!((1..=6).map(|k| node(&c, &format!("deep:c{k}"))).collect::<Vec<_>>(), [1, 2, 3, 4, 5, 6]);
        assert_eq!(node(&c, "wide:a"), 3);
        assert_eq!((2..8).map(|k| node(&c, &format!("wide:n{k}"))).collect::<Vec<_>>(), [6, 6, 5, 5, 4, 4]);
    }

    /// The re-levelled overlay compiled: each part's output position decodes what the
    /// part's circuit gives on the input, at every input byte. The parts share the one
    /// input slot, and its gather writes the channels both parts' input wires hold.
    #[test]
    fn a_relevelled_overlay_decodes_the_circuits_bytes() {
        use crate::llama::{tok, Model};
        let before = two_parts();
        let mut c = two_parts();
        assert_eq!(level(&mut c, &[(1, 6)], &[0; 7], &[0, 0], 3, 0, usize::MAX), FITS);
        assert_ne!(c.levels, before.levels);
        let m = Model::compile(&c);
        let cfg = &m.cfg;
        assert_eq!((cfg.layers, cfg.n_inputs, cfg.n_outputs), (7, 1, 2));
        assert_eq!((c.inputs.len(), c.held.len(), &c.parts), (1, 1, &vec![vec![0], vec![0]]), "the two parts hold input slot 0");
        let (mut deeps, mut wides) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
        let decode = |v: u8, at: usize| tok::output_value(cfg, m.forward(&[tok::input(cfg, 0, v), tok::output_mask(cfg, 1), tok::output_mask(cfg, 2)], false)[at]);
        for v in 0..=255u8 {
            let want = before.eval(&BTreeMap::from([("deep:X".to_string(), v), ("wide:X".to_string(), v)]));
            assert_eq!((decode(v, 1), decode(v, 2)), (Some(want["deep:Y"]), Some(want["wide:Y"])), "input {v:#04x}");
            deeps.insert(want["deep:Y"]);
            wides.insert(want["wide:Y"]);
        }
        // The chain's output bit takes both values and the wide part's six bits every
        // combination, so no output is right only because it is constant.
        assert_eq!((deeps.len(), wides.len()), (2, 64));
    }

    #[test]
    fn a_depth_too_few_for_a_part_is_refused() {
        let mut c = two_parts();
        assert_eq!(level(&mut c, &[(0, 5)], &[0; 7], &[0, 0], 3, 0, usize::MAX), None);
        assert_eq!(c.units_per_level(), vec![2, 7, 1, 1, 1, 1]);
    }

    #[test]
    fn a_node_with_no_later_level_takes_its_layer_and_the_overflow_is_reported() {
        // The chain fills a cap of 1, so nothing of the wide part fits anywhere: each
        // node lands at its earliest level, and level 2 ends six units over.
        let mut c = two_parts();
        assert_eq!(level(&mut c, &[(1, 6)], &[0; 7], &[0, 0], 1, 0, usize::MAX), Some(Over { units: 6, heads: 0 }));
        assert_eq!(c.units_per_level(), vec![2, 7, 1, 1, 1, 1]);
        // What a layer holds besides nodes counts against the cap too.
        let mut c = two_parts();
        assert_eq!(level(&mut c, &[(1, 6)], &[0, 0, 0, 0, 0, 2, 0], &[0, 0], 3, 0, usize::MAX), Some(Over { units: 0, heads: 0 }));
        assert_eq!(c.units_per_level(), vec![1, 2, 3, 3, 3, 1]);
    }

    #[test]
    fn a_part_at_its_live_limit_keeps_its_nodes_for_lower_levels() {
        // The wide part starts with its six outputs live. Its first node placed adds a
        // live wire below it (`a` and its own input bit for its output), so at a limit
        // of six each waits for its earliest level, where the cap no longer counts.
        let mut c = two_parts();
        assert_eq!(level(&mut c, &[(1, 6)], &[0; 7], &[0, 0], 3, 0, 6), Some(Over { units: 4, heads: 0 }));
        assert_eq!(c.units_per_level(), vec![2, 7, 1, 1, 1, 1]);
        // At seven the six nodes pack under the cap, and `a`, whose two input bits would
        // make eight, waits for its earliest level.
        let mut c = two_parts();
        assert_eq!(level(&mut c, &[(1, 6)], &[0; 7], &[0, 0], 3, 0, 7), FITS);
        assert_eq!((c.units_per_level(), node(&c, "wide:a")), (vec![2, 1, 1, 3, 3, 3], 1));
        // At eight the part packs as it does with no limit.
        let mut c = two_parts();
        assert_eq!(level(&mut c, &[(1, 6)], &[0; 7], &[0, 0], 3, 0, 8), FITS);
        assert_eq!((c.units_per_level(), node(&c, "wide:a")), (vec![1, 1, 2, 3, 3, 3], 3));
    }

    #[test]
    fn a_head_moves_with_its_query_and_its_readers() {
        let mut c = side_by_side(&[("deep", &chain(5)), ("mem", &head_part())]);
        assert_eq!((c.heads[0].layer, c.part_depth(1)), (1, 2));
        // The reader ends at the part's depth, the head a layer below it, and the node its
        // query reads at the head's own level.
        assert_eq!(level(&mut c, &[(1, 5)], &[0; 6], &[0, 0], 8, 1, usize::MAX), FITS);
        assert_eq!((c.heads[0].layer, node(&c, "mem:a"), node(&c, "mem:r")), (4, 4, 5));
        assert!(c.heads[0].out.iter().all(|&w| c.level(w) == 4));
    }

    #[test]
    fn the_heads_of_one_bus_move_as_one() {
        let part = bus_part();
        let g = Graph::of(&part);
        let groups: Vec<(usize, usize)> = (0..g.len()).filter(|&i| !g.is_node(i)).map(|i| g.weight(i)).collect();
        assert_eq!(groups, [(0, 2)], "two heads on one bus are one item of two heads");
        let mut c = side_by_side(&[("deep", &chain(6)), ("bus", &part)]);
        assert_eq!(c.heads.iter().map(|h| h.layer).collect::<Vec<_>>(), [2, 2]);
        // With room for two heads a layer both sit under the reader; `m0` alone could
        // have stayed a layer lower with `a`.
        assert_eq!(level(&mut c, &[(1, 6)], &[0; 7], &[0, 0], 8, 2, usize::MAX), FITS);
        assert_eq!(c.heads.iter().map(|h| h.layer).collect::<Vec<_>>(), [5, 5]);
        assert!(c.heads.iter().flat_map(|h| &h.out).all(|&w| c.level(w) == 5));
        assert_eq!((node(&c, "bus:a"), node(&c, "bus:b"), node(&c, "bus:r")), (4, 5, 6));
        // With room for one head a layer the pair has nowhere it fits, so it lands at its
        // earliest layer, one head over.
        let mut c = side_by_side(&[("deep", &chain(6)), ("bus", &part)]);
        assert_eq!(level(&mut c, &[(1, 6)], &[0; 7], &[0, 0], 8, 1, usize::MAX), Some(Over { units: 0, heads: 1 }));
        assert_eq!(c.heads.iter().map(|h| h.layer).collect::<Vec<_>>(), [2, 2]);
    }

    #[test]
    fn a_pinned_parts_heads_take_their_layers_room() {
        // Both parts' heads can only sit at layer 1. The pinned part's is there already,
        // so with room for one head the free part's lands over it.
        let part = head_part();
        let mut c = side_by_side(&[("held", &part), ("free", &part)]);
        assert_eq!(level(&mut c, &[(1, 2)], &[0; 3], &[0, 0], 8, 1, usize::MAX), Some(Over { units: 0, heads: 1 }));
        assert_eq!(c.heads.iter().map(|h| h.layer).collect::<Vec<_>>(), [1, 1]);
        let mut c = side_by_side(&[("held", &part), ("free", &part)]);
        assert_eq!(level(&mut c, &[(1, 2)], &[0; 3], &[0, 0], 8, 2, usize::MAX), FITS);
    }
}
