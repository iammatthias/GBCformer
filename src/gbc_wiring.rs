//! The Game Boy's side of the wiring table (`crate::wiring`): each stream's slots, flags
//! and codes read from the parts' slot ABIs, and the snapshot of a session at its
//! hand-off. Each entry comes from what the full program runs: a constant its own decoder
//! reads (`sm83::bits`, `sm83::mem_passes`, `control_circuit`'s and `apu_circuit`'s bit
//! names, `clock::DIGITS`, `machine::apu::GAIN`), or a probe of its function (the port codes
//! from `layout::port`, the span stream's event bytes from `Ctl::encode`, the footer from
//! `Rtc::footer`, the colours from `rgb`). The tests hold every slot name to its stream
//! and the presentation entries to the host's functions.

use crate::apu_circuit::{self, fields};
use crate::control_circuit::{self, circuit as ctl, clock, Ctl, Event, Span};
use crate::checkpoint::SlotAbi;
use crate::machine::fifo::Window;
use crate::machine::ppu::{rgb, DMG_SHADES};
use crate::memory::image::Image;
use crate::memory::layout::{self, Port, BLOCK, BLOCKED, CRAM, FF, HOST, IO, IO_BLOCKS, OAM, OAM_LEN, PAL, VRAM, VRAM_LEN};
use crate::one_model::Part;
use crate::ppu_circuit::circuit as ppu;
use crate::reference::Reference;
use crate::session::Session;
use crate::sm83::{bits, InMem, OutMem, Rq};
use crate::wiring::{Length, Stepper, Board, Console, PowerOn, Sparse, Code, Composed, Field, Flag, Lines, MemoryFile, Part as In, Ports, Present, Segments, Snapshot, Spans, Steps, Target, Wiring};

/// The header bytes that name the cartridge's board and its RAM, as `rom::identify` reads them.
const HEADER_TYPE: usize = 0x147;
const HEADER_RAM: usize = 0x149;

fn names(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|s| s.to_string()).collect()
}

fn out(o: OutMem) -> &'static str {
    OutMem::NAMES[o as usize]
}

fn field3(a: OutMem, b: OutMem, c: OutMem) -> Field {
    Field::of(&[out(a), out(b), out(c)])
}

/// The span stream's event bytes (`EV`, `EA`) for `event`, as `Ctl::encode` writes them.
fn event_code(event: Event) -> Code {
    let mut slots = [0u8; control_circuit::INPUTS.len()];
    Ctl::encode(&[0; control_circuit::INPUTS.len() - ctl::EVENTS.len() - ctl::OWED.len() - 3].try_into().unwrap(), &Span { mcycles: 0, since: 0, paused: false, event, cable: control_circuit::CABLE_IDLE, step: false, owed: 0 }, &mut slots);
    let at = |name: &str| control_circuit::INPUTS.iter().position(|n| *n == name).unwrap();
    [slots[at("EV")], slots[at("EA")]]
}

fn steps() -> Steps {
    use OutMem as O;
    let req = |r: Rq| Rq::NAMES[r as usize];
    let carry = [InMem::A, InMem::F, InMem::B, InMem::C, InMem::D, InMem::E, InMem::H, InMem::L, InMem::SPL, InMem::SPH, InMem::PCL, InMem::PCH].map(|i| (format!("{}'", InMem::NAMES[i as usize]), InMem::NAMES[i as usize].to_string()));
    // The passes as the memory core runs them without a lockstep (`sm83::mem_passes`).
    let slot = |p: usize| if p < Rq::NAMES.len() { Rq::NAMES[p] } else { OutMem::NAMES[p - Rq::NAMES.len()] }.to_string();
    let [step, result, fp, dma, wd, oam] = crate::sm83::mem_passes(false).map(|ps| ps.into_iter().map(slot).collect::<Vec<String>>());
    let unread: Vec<String> = crate::sm83::AUDIT_SLOTS.map(out).iter().map(|s| s.to_string()).collect();
    let rq = |m: u8| Flag::of(req(Rq::RQ), m);
    let wq = |m: u8| Flag::of(out(O::WQ), m);
    Steps {
        stream: Part::Cpu as usize,
        carry: carry.to_vec(),
        composed: vec![
            Composed { slot: "CTL".into(), parts: vec![In::Carry { from: out(O::CTL).into(), mask: bits::CTL_CARRIED }, In::Busy { bit: bits::CTL_DMA }, In::Locks { present: bits::CTL_LOCKS_PRESENT, mask: bits::CTL_LOCKS_MASK, shift: bits::CTL_LOCKS_SHIFT }] },
            Composed { slot: "IE".into(), parts: vec![In::State { slot: "IE".into() }] },
            Composed { slot: "IF".into(), parts: vec![In::State { slot: "IF".into() }] },
            Composed { slot: InMem::NAMES[InMem::DB as usize].into(), parts: vec![In::Emitted { slot: "DB".into() }] },
        ],
        step,
        result,
        fetch: fp,
        line: dma,
        second: wd,
        unread,
        late: [InMem::D0, InMem::D1, InMem::LK].map(|i| InMem::NAMES[i as usize].to_string()).to_vec(),
        served: bits::RQ_SERVED.map(rq).to_vec(),
        locks: rq(bits::RQ_LOCKS),
        locks_mask: bits::CTL_LOCKS_MASK,
        fetched: bits::RQ_FETCH.map(rq).to_vec(),
        read_at: vec![field3(O::RP0a, O::RP0b, O::RP0c), field3(O::RP1a, O::RP1b, O::RP1c)],
        fetch_at: vec![field3(O::FP0a, O::FP0b, O::FP0c), field3(O::FP1a, O::FP1b, O::FP1c), field3(O::FP2a, O::FP2b, O::FP2c)],
        reads: bits::RQ_READ.map(rq).to_vec(),
        host_block: HOST,
        host_mask: BLOCK - 1,
        writes: bits::WQ_WRITE.map(wq).to_vec(),
        value: names(&[out(O::WV0), out(O::WV1)]),
        write_at: vec![field3(O::WP0a, O::WP0b, O::WP0c), field3(O::WP1a, O::WP1b, O::WP1c)],
        records: bits::WQ_RECORD.map(wq).to_vec(),
        seconds: bits::WQ_SECOND.map(wq).to_vec(),
        second_at: vec![field3(O::WD0a, O::WD0b, O::WD0c), field3(O::WD1a, O::WD1b, O::WD1c)],
        access_at: [Rq::RC, Rq::WC].iter().flat_map(|&r| bits::CYCLE_SHIFT.map(|shift| Field::bits(req(r), shift, bits::CYCLE_MASK as u64))).collect(),
        access_locks: [Rq::RC, Rq::RC, Rq::RC, Rq::RC, Rq::WC, Rq::WC].map(|r| Flag::of(req(r), if r == Rq::RC { bits::RC_AT_LOCK } else { bits::WC_FETCH_LOCK })).to_vec(),
        fetch_cycles: vec![(Flag::of(req(Rq::RC), bits::FETCHED), 1), (Flag::of(req(Rq::WC), bits::FETCHED), 2)],
        access_in: names(&[InMem::NAMES[InMem::AS0 as usize], InMem::NAMES[InMem::AS1 as usize], InMem::NAMES[InMem::FS as usize]]),
        access_shift: crate::step::ACCESS_SHIFT,
        replace: crate::step::OVERRIDE,
        locks_bits: crate::step::VRAM_LOCKED | crate::step::OAM_LOCKED,
        busy_bit: crate::step::DMA_BLOCKING,
        write_slots: crate::step::WRITE_SLOTS,
        write_shift: crate::step::WRITE_SHIFT,
        ack: Field::bits(out(O::ACK), 0, bits::ACK_IRQ as u64),
        cycles: Field::bits(out(O::MC), 0, bits::MC_CYCLES as u64),
        halted: Flag::of(out(O::CTL), bits::CTL_HALTED),
        stop: Flag::of(out(O::CTL), bits::CTL_STOP),
        windows: vec![field3(O::MAP0a, O::MAP0b, O::MAP0c), field3(O::MAPRa, O::MAPRb, O::MAPRc), field3(O::MAPSa, O::MAPSb, O::MAPSc), field3(O::MAPWa, O::MAPWb, O::MAPWc)],
        line_in: [InMem::DSL, InMem::DSH, InMem::DDL, InMem::DDH].map(|i| InMem::NAMES[i as usize].to_string()).to_vec(),
        line_at: vec![field3(O::DPa, O::DPb, O::DPc), field3(O::DQa, O::DQb, O::DQc)],
        line_bits: layout::LINE_BITS as u8,
        fill_block: FF,
        fill_mask: BLOCK - 1,
        blocked: (BLOCKED, BLOCKED + BLOCK),
        save_window: crate::sm83::SAVE_WINDOW,
        save_regions: crate::sm83::SAVED.to_vec(),
        save_base: CRAM,
        events: events(oam),
        footer_len: footer().0,
        footer_bits: footer().1,
        time_at: footer().2,
    }
}

/// The OAM corruption bug's events (`step::step` runs them in the same order): the
/// fetches' (CTL' bits 5-7), the reads' (ACK bits 5-6), the idle M-cycles' (MC bits 3-4,
/// at MC bits 5-6), the writes' (WQ bits 6-7); the event byte `OB` with bit 0 for a read
/// (`sm83::oam_event_byte`); the OAM pass's slots, and its five rows' records
/// (`sm83::decode_oam`, `llama::tok::OAM_RECORDS` on).
fn events(pass: Vec<String>) -> crate::wiring::Events {
    use OutMem as O;
    let flags = |slot: O, masks: &[u8]| masks.iter().map(|&m| Flag::of(out(slot), m)).collect::<Vec<Flag>>();
    assert_eq!(crate::sm83::oam_event_byte(0, true), 1, "a read's event byte is the row with bit 0 set");
    crate::wiring::Events {
        fetches: flags(O::CTL, &bits::CTL_FE_FETCH),
        reads: flags(O::ACK, &bits::ACK_FE),
        idle: flags(O::MC, &bits::MC_IDLE),
        idle_at: Field::bits(out(O::MC), bits::MC_IDLE_SHIFT, bits::MC_IDLE_MASK as u64),
        writes: flags(O::WQ, &bits::WQ_FE),
        input: InMem::NAMES[InMem::OB as usize].to_string(),
        read_bit: 1,
        pass,
        rows: (0..5u8).map(|t| {
            let at = O::OW0a as usize + 3 * t as usize;
            (Flag::of(out(O::OT), 1 << t), crate::llama::tok::OAM_RECORDS + t, Field::of(&[O::NAMES[at], O::NAMES[at + 1], O::NAMES[at + 2]]))
        }).collect(),
    }
}

/// The clock's save footer as bit routes from the spans state, probed through the full
/// program's own `decode_state` and `Rtc::footer`, one state bit at a time.
fn footer() -> (usize, Vec<(String, u8, usize, u8)>, usize) {
    let names: Vec<&str> = ctl::STATE.iter().chain(&clock::STATE).copied().collect();
    let footer = |state: &control_circuit::State, time: u64| control_circuit::decode_state(state).clock.footer(Some(time));
    let zero = footer(&[0; ctl::STATE.len() + clock::STATE.len()], 0);
    assert!(zero.iter().all(|&b| b == 0), "a zero clock's footer is zero");
    let mut bits = Vec::new();
    for (slot, name) in names.iter().enumerate().skip(ctl::STATE.len()) {
        for bit in 0..8u8 {
            let mut s = [0u8; ctl::STATE.len() + clock::STATE.len()];
            s[slot] = 1 << bit;
            for (byte, &b) in footer(&s, 0).iter().enumerate() {
                for to in 0..8u8 {
                    if b >> to & 1 != 0 {
                        bits.push((name.to_string(), bit, byte, to));
                    }
                }
            }
        }
    }
    let time = footer(&[0; ctl::STATE.len() + clock::STATE.len()], u64::MAX);
    let time_at = time.iter().position(|&b| b == 0xFF).expect("the footer holds the time");
    assert!(time[time_at..].iter().all(|&b| b == 0xFF) && time_at + 8 == time.len(), "the time ends the footer");
    (zero.len(), bits, time_at)
}

fn spans() -> Spans {
    let f = |s: &str| Field::of(&[s]);
    let w = |a: &str, b: &str| Field::of(&[a, b]);
    let fx = |m: u8| Flag::of("FX", m);
    let carry = ctl::NEXT.iter().chain(&clock::NEXT).zip(ctl::STATE.iter().chain(&clock::STATE)).map(|(o, i)| (o.to_string(), i.to_string())).collect();
    Spans {
        stream: Part::Control as usize,
        carry,
        phase: "LM".into(),
        position: "LY".into(),
        cycles: names(&["D0", "D1"]),
        paused_cycles: names(&["Q0", "Q1"]),
        moved: names(&["H0", "H1"]),
        event: names(&["EV", "EA", "EX"]),
        fixed: vec![("SI".into(), control_circuit::CABLE_IDLE)],
        late: "ML".into(),
        request: names(&control_circuit::REQUEST),
        asks: Flag::of(control_circuit::REQUEST[0], control_circuit::MQ_LENGTH),
        tick: event_code(Event::Tick),
        ack: event_code(Event::Ack(0)),
        raise: event_code(Event::Raise(0)),
        buttons: event_code(Event::Buttons(0)),
        stop: event_code(Event::Stop),
        elapse: event_code(Event::Elapse(0)),
        digits: clock::DIGITS,
        rest: event_code(Event::Halt),
        wake_up: event_code(Event::Wake),
        step_end: event_code(Event::StepEnd),
        lock_query: event_code(Event::Read(0xFF41)),
        row_query: event_code(Event::Read(crate::machine::control::OAM_ROW_ADDR)),
        read: f("RV"),
        horizon: Field::masked(&["H0", "H1"], u64::MAX, 1),
        busy_left: f("DY"),
        busy_from: f("DZ"),
        serial_start: fx(ctl::FX_SERIAL_START),
        busy: fx(ctl::FX_OAM_DMA),
        render: fx(ctl::FX_RENDER),
        frame: fx(ctl::FX_FRAME),
        blank: fx(ctl::FX_LCD_OFF),
        pause: fx(ctl::FX_PAUSE),
        run: fx(ctl::FX_OAM_BYTE),
        run_page: f("DP"),
        run_line: Field::masked(&["DX"], !ctl::DX_LAST_EARLY as u64, 0),
        run_last_early: Flag::of("DX", ctl::DX_LAST_EARLY),
        run_before: f("DM"),
        run_last: f("DL"),
        run_dest: f("DD"),
        run_fill: f("DF"),
        run_bits: control_circuit::OAM_LINE_BITS,
        block_copy: fx(ctl::FX_BLOCK),
        block_source: w("CS0", "CS1"),
        block_dest: w("CD0", "CD1"),
        block_len: Field::masked(&["CL"], control_circuit::CL_LEN as u64, 0),
        block_held: Flag::of("CL", control_circuit::CL_HELD),
        step_input: ctl::STEP.into(),
        stall: w("SS0", "SS1"),
        owed: names(&ctl::OWED),
        own: Flag::of("PA", control_circuit::PA_PRESENT),
        own_index: Field::masked(&["PA"], control_circuit::PA_INDEX as u64, 0),
        first: Field::masked(&["RF"], u64::MAX, 1),
        count: f("RS"),
        period: f("RP"),
        tail: f("RT"),
        start_phase: f("RM"),
        lines: (0..ctl::SCHEDULE).map(|j| f(&format!("SL{j}"))).collect(),
        edge: Field::masked(&["FE0", "FE1"], control_circuit::FE_EDGE as u64, 1),
        edge_held: Flag::of("FE1", ctl::FE_HELD),
        edge_drop: Field::bits("FE1", ctl::FE_DROP_SHIFT, (ctl::FE_DROP >> ctl::FE_DROP_SHIFT) as u64),
        rise: Field::masked(&["FR0", "FR1"], control_circuit::FE_EDGE as u64, 1),
        wake: Flag::of("FE1", ctl::FE_WAKE),
        rate: f("HR"),
        fast_rate: control_circuit::FAST_RATE as u64,
        rates: control_circuit::RATES.map(u64::from).to_vec(),
        frame_time: w("FH0", "FH1"),
        locks: [(0, ["MX0", "MX1"]), (control_circuit::LK_SHIFT, ["MY0", "MY1"])].map(|(shift, span)| (Field::bits("LK", shift, control_circuit::LK_MASK as u64), Field::masked(&span, control_circuit::LOCK_SPAN as u64, 1))).to_vec(),
        follow: f("NE"),
        idle: Flag::of("HDMA5", 0x80),
        kept: Flag::of("FL", ctl::FL_HALT),
        uncounted: Flag::of("HDMA3", ctl::HP),
        held: Flag::of("TC", control_circuit::TC_BLOCKED),
        serial: "SB".into(),
        max_span: ctl::MAX_SPAN,
    }
}

fn segments() -> Segments {
    let carry = apu_circuit::RESULT[..fields::BYTES].iter().zip(&apu_circuit::INPUTS[..fields::BYTES]).map(|(o, i)| (o.to_string(), i.to_string())).collect();
    Segments {
        stream: Part::Apu as usize,
        carry,
        write_address: "WA".into(),
        write_value: "WV".into(),
        events: "EV".into(),
        on_write: apu_circuit::EV_WRITE,
        on_fall: apu_circuit::EV_FALL,
        on_reset: apu_circuit::EV_RESET,
        on_rise: apu_circuit::EV_RISE,
        on_sample: apu_circuit::EV_SAMPLE,
        on_fast: apu_circuit::EV_FAST,
        cycles: "D".into(),
        read_address: "RA".into(),
        sample: vec![Field::of(&["SL0", "SL1"]), Field::of(&["SR0", "SR1"])],
        read: "RV".into(),
        to_sample: "SD".into(),
        to_end: "SE".into(),
        left: "DL".into(),
        max_cycles: apu_circuit::MAX_DELTA as u64,
        max_fast_cycles: 2 * apu_circuit::MAX_DELTA as u64,
    }
}

fn lines() -> Lines {
    let flags = |armed, mode, high, blocked| ppu::inputs(0, Window { rows: 0, armed, carried: false }, mode, high, blocked)[2];
    use crate::machine::ppu::Mode;
    Lines {
        stream: Part::Ppu as usize,
        line: ppu::INPUTS[0].into(),
        counter: ppu::INPUTS[1].into(),
        flags: ppu::INPUTS[2].into(),
        armed: flags(true, Mode::Dmg, false, false),
        features: flags(false, Mode::Cgb, false, false),
        high: flags(false, Mode::Dmg, true, false),
        compat: flags(false, Mode::Compat, false, false),
        held: flags(false, Mode::Dmg, false, true),
        pixels: (0..crate::machine::ppu::WIDTH).map(|x| format!("PX{x}")).collect(),
        counter_out: ppu::WINDOW[0].into(),
        armed_out: ppu::WINDOW[1].into(),
        armed_mask: 1,
        carried: ppu::CARRIED_IN,
        carried_mask: ppu::CARRIED_OUT,
        length: ppu::LENGTH.into(),
        own_memory: PAL,
        mode_cell: crate::ppu_circuit::video(Default::default()).cells[0],
        mode_mask: 1,
        follows: crate::ppu_circuit::video(Default::default()).ranges,
    }
}

/// Each port code's target, from `layout::port` at every low address of its block.
fn ports() -> Ports {
    let low_bits = BLOCK.trailing_zeros() as u8;
    let mut codes = Vec::new();
    let mut indirect_reads = Vec::new();
    for code in 0..IO_BLOCKS {
        let at = |low: u32| layout::port(IO + code * BLOCK + low);
        let Some(first) = at(0) else { continue };
        let target = match first {
            Port::Control(_) | Port::Clock(_) => {
                let bytes = |e: &dyn Fn(u16) -> Event| -> (u8, Vec<u8>) {
                    let all: Vec<Code> = (0..BLOCK).map(|low| match at(low) {
                        Some(Port::Control(a) | Port::Clock(a)) => event_code(e(a)),
                        other => panic!("port code {code}: low {low} names {other:?}"),
                    }).collect();
                    assert!(all.iter().all(|c| c[0] == all[0][0]), "port code {code}: one event code per block");
                    (all[0][0], all.iter().map(|c| c[1]).collect())
                };
                let read = bytes(&|a| Event::Read(a));
                let write = bytes(&|a| Event::Write(a, 0));
                if let Port::Control(_) = first {
                    let palette: Vec<u8> = (0..BLOCK).filter(|&low| matches!(at(low), Some(Port::Control(0xFF69 | 0xFF6B)))).map(|low| low as u8).collect();
                    indirect_reads.push((code, palette));
                }
                Target::Spans { read: Some(read), write: Some(write) }
            }
            Port::Apu(_) => Target::Segments { address: (0..BLOCK).map(|low| match at(low) { Some(Port::Apu(a)) => a as u8, other => panic!("port code {code}: {other:?}") }).collect() },
            Port::Display(i) => Target::Own { index: i },
        };
        codes.push((code, target));
    }
    Ports { base: IO, low_bits, codes, indirect_reads }
}

fn length() -> Length {
    Length {
        stream: Part::Length as usize,
        line: ppu::LENGTH_INPUTS[0].into(),
        flags: ppu::LENGTH_INPUTS[1].into(),
        armed: ppu::length_inputs(0, Window { rows: 0, armed: true, carried: false }, crate::machine::ppu::Mode::Dmg, None)[1],
        carried: ppu::CARRIED_IN,
        length: ppu::LENGTH_OUTPUTS[0].into(),
        armed_out: ppu::LENGTH_OUTPUTS[1].into(),
        armed_mask: 1,
        carried_mask: ppu::CARRIED_OUT,
        skip: (VRAM, VRAM + VRAM_LEN as u32),
        write: [2, 3, 4].map(|i| ppu::LENGTH_INPUTS[i].into()),
        write_flag: ppu::LENGTH_WRITE,
        verdict: ppu::LENGTH_OUTPUTS[2].into(),
        marks: ppu::JUDGE_MARK,
        takes: ppu::JUDGE_TAKE,
        time: [3, 4].map(|i| ppu::LENGTH_OUTPUTS[i].into()),
    }
}

/// The stepper's slots and bits, as `stepper::Stepper::line` drives it.
fn stepper() -> Stepper {
    use crate::ppu_circuit::stepper as st;
    Stepper {
        stream: Part::Stepper as usize,
        line: st::LINE_INPUTS[0].into(),
        flags: st::LINE_INPUTS[1].into(),
        counter: st::LINE_INPUTS[2].into(),
        start: st::FLG_START,
        armed: st::FLG_ARMED,
        carried: st::FLG_CARRIED,
        fast: st::FLG_DOUBLE,
        held: st::FLG_BLOCKED,
        write: [3, 4, 5, 6, 7].map(|i| st::LINE_INPUTS[i].into()),
        pending: st::PN_PENDING,
        carry: st::state_slots().into_iter().map(|(s, _)| (format!("{s}'"), s)).collect(),
        report: st::RESULTS[0].into(),
        applied: st::OUT_APPLIED,
        finished: st::OUT_FINISHED,
        pixel: st::OUT_PIXEL,
        done: st::OUT_DONE,
        x: st::RESULTS[1].into(),
        colour: [2, 3].map(|i| st::RESULTS[i].into()),
        counter_out: st::RESULTS[4].into(),
        window_out: st::RESULTS[5].into(),
        armed_mask: st::WFLG_ARMED,
        carried_mask: st::WFLG_CARRIED,
        length: [6, 7].map(|i| st::RESULTS[i].into()),
        // The dots to mode 0 into the spans stream's unit, less `lcd::LENGTH_BASE`
        // (`stepper::length`).
        shift: 0,
        bias: -(crate::machine::lcd::LENGTH_BASE as i64),
    }
}

fn present() -> Present {
    // Each output byte (red, green, blue) is one 5-bit input field widened: probe which.
    let shift = |byte: u32| -> u8 { (0..3u8).map(|k| 5 * k).find(|&s| rgb(true, 31 << s) >> byte & 0xFF == 0xFF).expect("a channel") };
    let shifts = [16, 8, 0].map(shift);
    let widen: Vec<u8> = (0..32u16).map(|c| (rgb(true, c << shifts[0]) >> 16) as u8).collect();
    Present {
        width: crate::machine::ppu::WIDTH,
        height: crate::machine::ppu::HEIGHT,
        shades: DMG_SHADES.to_vec(),
        channel_shifts: shifts.to_vec(),
        widen,
        blank_shade: DMG_SHADES[0],
        blank_colour: rgb(true, 0x7FFF),
        gain: crate::machine::apu::GAIN,
        shift: crate::machine::apu::SHIFT,
        buttons: ["right", "left", "up", "down", "a", "b", "select", "start"].iter().map(|n| (n.to_string(), crate::machine::io::Buttons::parse(n).expect("a button"))).collect(),
    }
}

/// The one model's wiring table.
pub fn wiring() -> Wiring {
    Wiring { steps: steps(), spans: spans(), segments: segments(), lines: lines(), length: length(), stepper: stepper(), ports: ports(), present: present(), power_on: power_on() }
}

/// A session at its hand-off as data: the snapshot, the memory image, the lines stream's
/// own image and the two frames, the ROM and boot ROM still in the images.
fn hand_off(warm: &Session<Reference>) -> Result<(Snapshot, Image, Image, Image, Vec<u32>), String> {
    let machine = warm.machine();
    let memory = warm.memory_image()?;
    let (display, window) = warm.ppu_state();
    // The lines stream's own image, as `PpuCheckpoint::new` builds it.
    let mut own = Image::new(&[memory.read(0)], 0, memory.banking()).with_palettes();
    own.fill(VRAM, memory.region(VRAM, VRAM_LEN));
    own.fill(OAM, memory.region(OAM, OAM_LEN));
    own.fill(PAL, &display);
    // The length stream's image, as `LengthCheckpoint::new` builds it: no VRAM.
    let mut len = Image::new(&[memory.read(0)], 0, memory.banking()).with_palettes();
    len.fill(OAM, memory.region(OAM, OAM_LEN));
    len.fill(PAL, &display);
    let file = |name: &str, image: &Image| MemoryFile { bytes: format!("{name}.bin"), present: format!("{name}.present"), lengths: image.parts().2 };
    let cpu = warm.cpu();
    let mut steps: Vec<(String, u8)> = [cpu.a, cpu.f, cpu.b, cpu.c, cpu.d, cpu.e, cpu.h, cpu.l, cpu.sp as u8, (cpu.sp >> 8) as u8, cpu.pc as u8, (cpu.pc >> 8) as u8].into_iter().zip(InMem::NAMES).map(|(b, n)| (n.to_string(), b)).collect();
    steps.push(("CTL".into(), cpu.ime as u8 | (cpu.ei_pending as u8) << 1 | (cpu.halted as u8) << 2 | (cpu.halt_bug as u8) << 3));
    let control = warm.control_reference().ok_or("the control logic was already handed to a device")?;
    let state = control_circuit::encode_state(&control.state());
    let spans = ctl::STATE.iter().chain(&clock::STATE).zip(state).map(|(n, b)| (n.to_string(), b)).collect();
    let apu = warm.apu_reference().ok_or("the APU was already handed to a device")?;
    let segments = apu_circuit::INPUTS[..fields::BYTES].iter().zip(fields::state_of(apu)).map(|(n, b)| (n.to_string(), b)).collect();
    // The modes a line draws in with the COMPAT cell clear and set, as `PpuCheckpoint::mode` picks them.
    let cgb = machine.hardware() == crate::machine::Hardware::Cgb;
    use crate::machine::ppu::Mode;
    let modes = if cgb { [Mode::Cgb, Mode::Compat] } else { [Mode::Dmg, Mode::Dmg] };
    let line_modes = modes.map(|m| ppu::inputs(0, Window::default(), m, false, false)[2]);
    let line_mode = memory.banking().get(layout::Reg::Compat) & 1 != 0;
    let (battery, clock) = machine.cart_facts();
    let (back, front) = machine.ppu.frames();
    let frames = back.iter().chain(front.iter()).copied().collect();
    let snapshot = Snapshot {
        steps,
        spans,
        segments,
        window: (window.rows, window.armed, window.carried),
        line_modes,
        line_mode,
        line_held: machine.ppu.held(),
        line_passes: if cgb { 2 } else { 1 },
        colours: cgb,
        save_len: if battery { machine.cart_ram_len() } else { 0 },
        save_pending: machine.pending_save(),
        clock,
        length: warm.ppu_lengths().latest(),
        frame: warm.frame_number(),
        overshoot: warm.overshoot(),
        clock_reading: machine.host_clocks().0,
        filter: machine.host_clocks().1,
        memory: file("memory", &memory),
        own: file("own", &own),
        len: file("len", &len),
        length_modes: modes.map(|m| ppu::length_inputs(0, Window::default(), m, None)[1]),
        stepper_modes: modes.map(|mode| crate::ppu_circuit::stepper::flags(crate::ppu_circuit::stepper::Constants { ly: 0, mode, double: false, blocked: false })),
        rom: (layout::ROM, memory.rom_len()),
        own_rom: (layout::ROM, 1),
        boot: machine.boot_len().map(|n| (layout::BOOT, n)),
        frames: "frames.bin".into(),
        wiring_sha1: String::new(),
        tensors_sha1: String::new(),
    };
    Ok((snapshot, memory, own, len, frames))
}

/// The digest of this build's wiring (`wiring::Manifest::wiring_sha1`).
pub fn wiring_sha1() -> String {
    crate::wiring::sha1(&serde_json::to_string(&wiring()).expect("the wiring serializes"))
}

/// `GBCFORMER_SNAPSHOT=<dir>`: the session at its hand-off as data, `snapshot.json`,
/// `memory.bin`, `memory.present`, `own.bin`, `own.present` and `frames.bin`, with the
/// ROM and boot ROM left out: the runner fills them in.
pub fn write_snapshot(warm: &Session<Reference>, dir: &std::path::Path) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{}: {e}", dir.display());
    std::fs::create_dir_all(dir).map_err(io)?;
    let (mut snapshot, memory, own, len, frames) = hand_off(warm)?;
    snapshot.wiring_sha1 = wiring_sha1();
    if let Some(path) = std::env::var_os("GBCFORMER_GGUF_ONLY") {
        snapshot.tensors_sha1 = crate::wiring::Manifest::read(std::path::Path::new(&path))?.0.tensors_sha1;
    }
    let write = |f: &MemoryFile, image: &Image, blank: &[(u32, usize)]| -> Result<(), String> {
        let (bytes, present, _) = image.parts();
        let mut bytes = bytes.to_vec();
        for &(base, len) in blank {
            bytes[base as usize..base as usize + len].fill(0);
        }
        std::fs::write(dir.join(&f.bytes), bytes).map_err(io)?;
        std::fs::write(dir.join(&f.present), present.iter().map(|&p| p as u8).collect::<Vec<u8>>()).map_err(io)
    };
    write(&snapshot.memory, &memory, &[Some(snapshot.rom), snapshot.boot].into_iter().flatten().collect::<Vec<_>>())?;
    write(&snapshot.own, &own, &[snapshot.own_rom])?;
    write(&snapshot.len, &len, &[snapshot.own_rom])?;
    std::fs::write(dir.join(&snapshot.frames), frames.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>()).map_err(io)?;
    std::fs::write(dir.join("snapshot.json"), serde_json::to_string(&snapshot).map_err(|e| e.to_string())?).map_err(io)
}

/// An image as its nonzero bytes and its runs of addresses with a token.
fn sparse(image: &Image) -> Sparse {
    let (bytes, present, lengths) = image.parts();
    let bytes = bytes.iter().enumerate().filter(|(_, b)| **b != 0).map(|(p, &b)| (p as u32, b)).collect();
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for (p, &here) in present.iter().enumerate() {
        match runs.last_mut() {
            Some((_, end)) if here && *end == p as u32 => *end += 1,
            _ if here => runs.push((p as u32, p as u32 + 1)),
            _ => {}
        }
    }
    Sparse { len: present.len(), bytes, present: runs, lengths }
}

/// The power-on states a runner builds from a ROM and a boot ROM alone: per console, the
/// hand-off of a session at power-on with a blank 32 KiB ROM-only cartridge (its ROM's
/// cells and its registers then left out), and every board the header can name, read
/// from sessions at power-on with a cartridge of that header and length.
pub fn power_on() -> PowerOn {
    use crate::machine::Hardware;
    let header = |kind: u8, ram: u8, len: usize| -> Vec<u8> {
        let mut rom = vec![0u8; len];
        rom[HEADER_TYPE] = kind;
        rom[HEADER_RAM] = ram;
        rom
    };
    let consoles = [(Hardware::Dmg, 0x100usize), (Hardware::Cgb, 0x900)]
        .into_iter()
        .map(|(hardware, boot_len)| {
            let session = |rom: Vec<u8>| Session::power_on(rom, None, hardware, vec![0; boot_len]);
            let warm = session(header(0, 0, 0x8000)).expect("a blank cartridge powers on");
            let (snapshot, mut memory, own, len, frames) = hand_off(&warm).expect("a power-on hand-off");
            let regs = (layout::REGS, layout::N_REGS);
            let mut len = len;
            len.fill(regs.0, &[0; layout::N_REGS]);
            memory.fill(regs.0, &[0; layout::N_REGS]);
            memory.fill(snapshot.rom.0, &vec![0; snapshot.rom.1]);
            let mut own = own;
            own.fill(regs.0, &[0; layout::N_REGS]);
            let mut boards = Vec::new();
            for kind in 0..=255u8 {
                for ram in 0..=255u8 {
                    // A header the host refuses at 32 KiB it refuses at every length.
                    if session(header(kind, ram, 0x8000)).is_err() {
                        continue;
                    }
                    for len in (15..=23).map(|b| 1usize << b) {
                        let Ok(s) = session(header(kind, ram, len)) else { continue };
                        let Ok(image) = s.memory_image() else { continue };
                        let ram_len = s.machine().cart_ram_len();
                        let (battery, clock) = s.machine().cart_facts();
                        boards.push(Board { kind, ram, rom_len: len, regs: image.region(regs.0, regs.1).to_vec(), ram_len, ram_fill: if ram_len > 0 { image.read(CRAM) } else { 0 }, save_len: if battery { ram_len } else { 0 }, clock });
                    }
                }
            }
            // Which offsets of the boot image the boot ROM gives: a boot ROM of 0xFF over a ROM of 0.
            let probe = Session::power_on(header(0, 0, 0x8000), None, hardware, vec![0xFF; boot_len]).expect("a blank cartridge powers on");
            let (_, image, _, _, _) = hand_off(&probe).expect("a power-on hand-off");
            let boot = snapshot.boot.expect("a power-on snapshot maps the boot ROM");
            let mut boot_overlay: Vec<(usize, usize)> = Vec::new();
            for (k, &b) in image.region(boot.0, boot.1).iter().enumerate() {
                match boot_overlay.last_mut() {
                    Some((_, end)) if b == 0xFF && *end == k => *end += 1,
                    _ if b == 0xFF => boot_overlay.push((k, k + 1)),
                    _ => {}
                }
            }
            Console { boot_len, boot_overlay, snapshot, memory: sparse(&memory), own: sparse(&own), len: sparse(&len), frames, boards }
        })
        .collect();
    PowerOn { header_type: HEADER_TYPE, header_ram: HEADER_RAM, regs: layout::REGS, ram: CRAM, consoles }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each saved region is a power-of-two block aligned to its length, so the battery
    /// tracker's test of a write's address reads a field of the address the model emitted
    /// (its high bits), as a port's dispatch does, and names no range of its own; the save's
    /// bytes start at the first.
    #[test]
    fn the_saved_regions_are_fields_of_an_emitted_address() {
        let s = wiring().steps;
        assert_eq!(s.save_regions[0].0, s.save_base);
        for &(a, b) in &s.save_regions {
            let len = b - a;
            assert!(len.is_power_of_two() && a % len == 0, "{a:#x}..{b:#x}");
        }
    }

    #[test]
    fn the_wiring_round_trips_through_json() {
        let w = wiring();
        let back: Wiring = serde_json::from_str(&serde_json::to_string(&w).unwrap()).unwrap();
        assert_eq!(back, w);
    }

    /// Every slot name the table holds is a slot of the stream that reads it, so a slot
    /// the compiler renames or drops fails here rather than in the runner. The steps
    /// stream's references into the spans stream's state (a composed input's state byte,
    /// the clock footer's bits) are held to the spans stream.
    #[test]
    fn every_slot_name_is_a_slot_of_its_stream() {
        let (_, specs) = crate::one_model::circuit(&crate::one_model::parts());
        let w = serde_json::to_value(wiring()).unwrap();
        let slots = |part: Part| -> std::collections::BTreeSet<String> { specs[part as usize].slots.iter().cloned().collect() };
        fn walk(v: &serde_json::Value, path: &str, out: &mut Vec<(String, String)>) {
            match v {
                serde_json::Value::String(s) => out.push((path.to_string(), s.clone())),
                serde_json::Value::Array(xs) => xs.iter().for_each(|x| walk(x, path, out)),
                serde_json::Value::Object(m) => m.iter().for_each(|(k, x)| walk(x, &format!("{path}.{k}"), out)),
                _ => {}
            }
        }
        let mut checked = 0;
        for (section, part) in [("steps", Part::Cpu), ("spans", Part::Control), ("segments", Part::Apu), ("lines", Part::Ppu), ("length", Part::Length), ("stepper", Part::Stepper)] {
            let mut names = Vec::new();
            walk(&w[section], section, &mut names);
            let (own, spans) = (slots(part), slots(Part::Control));
            for (path, name) in names {
                let into_spans = path.ends_with(".State.slot") || path.ends_with(".footer_bits");
                let set = if into_spans { &spans } else { &own };
                assert!(set.contains(&name), "{path}: \"{name}\" is no slot of the {} stream", if into_spans { "spans" } else { section });
                checked += 1;
            }
        }
        assert!(checked > 300, "{checked} slot names checked");
    }

    /// The runner's filter, from `Present` alone, is the board's filter.
    #[test]
    fn the_present_filter_is_the_boards() {
        let p = present();
        let mut board = crate::machine::apu::board();
        let mut cap = [0i32; 2];
        let mut x = 12345u32;
        for _ in 0..20000 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let mix = [(x >> 8) as i16 % 481, (x >> 20) as i16 % 481 - 240];
            let mut want = [0i16; 2];
            for side in 0..2 {
                let input = mix[side] as i32 * p.gain;
                let f = input - (cap[side] >> p.shift);
                cap[side] += f;
                want[side] = f.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
            assert_eq!(board.apply(mix), want);
        }
    }

    /// The presentation tables give the host's colours for every 15-bit colour and shade.
    #[test]
    fn the_present_colours_are_the_hosts() {
        let p = present();
        for c in 0..0x8000u16 {
            let got = p.channel_shifts.iter().fold(0u32, |rgb, &s| rgb << 8 | p.widen[(c >> s & 31) as usize] as u32);
            assert_eq!(got, rgb(true, c), "colour {c:#06x}");
        }
        for c in 0..4u16 {
            assert_eq!(p.shades[c as usize], rgb(false, c));
        }
    }

    /// A wiring with a field this build does not know (the `reread` flag the steps role
    /// carried in schema 1) is refused, not ignored.
    #[test]
    fn a_wiring_with_a_field_this_build_does_not_know_is_refused() {
        let mut v = serde_json::to_value(wiring()).unwrap();
        v["steps"]["reread"] = serde_json::json!({ "slot": "CTL'", "mask": 8 });
        let e = serde_json::from_value::<crate::wiring::Wiring>(v).unwrap_err().to_string();
        assert!(e.contains("unknown field `reread`"), "{e}");
    }

    /// A snapshot written for another wiring, or by another model, is refused by the manifest.
    #[test]
    fn a_snapshot_for_another_wiring_is_refused() {
        let w = wiring();
        let mut snap = w.power_on.consoles[0].snapshot.clone();
        let m = crate::wiring::Manifest { wiring_version: crate::wiring::WIRING_VERSION, wiring_sha1: wiring_sha1(), tensors_sha1: "1".repeat(40), vocab: 0, llama_cpp: String::new() };
        assert!(m.check_snapshot(&snap).is_err(), "a power-on snapshot names no wiring");
        (snap.wiring_sha1, snap.tensors_sha1) = (wiring_sha1(), "1".repeat(40));
        assert_eq!(m.check_snapshot(&snap), Ok(()));
        snap.tensors_sha1 = "2".repeat(40);
        assert!(m.check_snapshot(&snap).unwrap_err().starts_with("the snapshot was written by the model 2222"));
        snap.wiring_sha1 = "0".repeat(40);
        assert!(m.check_snapshot(&snap).unwrap_err().starts_with("the snapshot was written for wiring 0000"));
    }
}

#[cfg(test)]
mod footer_tests {
    use super::*;

    /// The routed footer is `Rtc::footer` over random clock states.
    #[test]
    fn the_footer_routes_are_the_clocks_footer() {
        let (len, bits, time_at) = footer();
        let names: Vec<&str> = ctl::STATE.iter().chain(&clock::STATE).copied().collect();
        let mut x = 99u32;
        for _ in 0..2000 {
            let mut s = [0u8; ctl::STATE.len() + clock::STATE.len()];
            for b in &mut s[ctl::STATE.len()..] {
                x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (x >> 24) as u8;
            }
            let want = control_circuit::decode_state(&s).clock.footer(Some(x as u64));
            let mut got = vec![0u8; len];
            for (slot, bit, byte, to) in &bits {
                let v = s[names.iter().position(|n| n == slot).unwrap()];
                got[*byte] |= (v >> bit & 1) << to;
            }
            got[time_at..].copy_from_slice(&(x as u64).to_le_bytes());
            assert_eq!(got, want.to_vec());
        }
    }

}
