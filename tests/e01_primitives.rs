//! E01: exact primitives through the real checkpoint. An 8-bit add-with-carry and its
//! flags, checked exhaustively in the IR and through the dense Llama forward pass.

use gbcformer::circuit::{byte, value, Circuit, Lin};
use gbcformer::llama::{tok, Model};
use std::collections::BTreeMap;

fn adder() -> Circuit {
    let mut c = Circuit::default();
    let a = byte(c.input("A"));
    let b = byte(c.input("B"));
    let cin = byte(c.input("CIN"));
    let s = value(&a, 8) + value(&b, 8) + cin[0].clone();
    let (sum, carry, half) = c.add("add", &a, &b, cin[0].clone(), 8);
    let sum: [Lin; 8] = sum.try_into().unwrap();
    let zero = c.node("zero", vec![
        gbcformer::circuit::Step { coef: 1, lin: s.clone(), theta: 0 },
        gbcformer::circuit::Step { coef: -1, lin: s.clone(), theta: 1 },
        gbcformer::circuit::Step { coef: 1, lin: s.clone(), theta: 256 },
        gbcformer::circuit::Step { coef: -1, lin: s, theta: 257 },
    ]);
    let k0 = || Lin::konst(0);
    c.output("SUM", sum);
    c.output("FLAGS", [k0(), k0(), k0(), k0(), carry, half, k0(), zero]);
    c
}

fn expect(a: u8, b: u8, cin: u8) -> (u8, u8) {
    let s = a as u16 + b as u16 + cin as u16;
    let h = (a & 15) + (b & 15) + cin >= 16;
    let f = ((s > 255) as u8) << 4 | (h as u8) << 5 | (((s & 255) == 0) as u8) << 7;
    (s as u8, f)
}

fn inputs(a: u8, b: u8, cin: u8) -> BTreeMap<String, u8> {
    [("A".into(), a), ("B".into(), b), ("CIN".into(), cin)].into()
}

#[test]
fn adder_ir_exhaustive() {
    let c = adder();
    for a in 0..=255u8 {
        for b in 0..=255u8 {
            for cin in 0..2 {
                let out = c.eval(&inputs(a, b, cin));
                assert_eq!((out["SUM"], out["FLAGS"]), expect(a, b, cin), "{a}+{b}+{cin}");
            }
        }
    }
}

#[test]
fn adder_checkpoint_exhaustive() {
    let c = adder();
    let m = Model::compile(&c);
    let cfg = &m.cfg;
    eprintln!("hidden {} inter {} layers {} vocab {}", cfg.hidden, cfg.intermediate, cfg.layers, cfg.vocab);
    for a in 0..=255u8 {
        for b in 0..=255u8 {
            for cin in 0..2 {
                let tokens = [tok::input(cfg, 0, a), tok::input(cfg, 1, b), tok::input(cfg, 2, cin), tok::output_mask(cfg, 3), tok::output_mask(cfg, 4)];
                for causal in [false, true] {
                    let out = m.forward(&tokens, causal);
                    let got = (tok::output_value(cfg, out[3]), tok::output_value(cfg, out[4]));
                    let (s, f) = expect(a, b, cin);
                    assert_eq!(got, (Some(s), Some(f)), "{a}+{b}+{cin} causal={causal}");
                }
            }
        }
    }
}

