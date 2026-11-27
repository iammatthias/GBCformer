//! The quarantine: the runner's source, and the transport it shares with the full program
//! (`src/transport.rs` without its tests), hold no rule of the machine the model computes.
//! With comments stripped they name no slot of the model's streams, hold no hexadecimal
//! literal (no address, mask or register), only decimal literals from a short list of
//! units, and none of the machine's terms; and every line of theirs that branches or picks
//! (`if`, `match`, a loop, a filter, a minimum) is one a review classified, listed in
//! `tests/branches.txt` with its class.

use std::path::Path;

/// Decimal literals the runner may hold: small counts (indices, a default hold of 3 frames,
/// a pixel's 4 bytes, a byte's 8 bits), 60 frames to the scripted clock's second, and
/// 1000 calls to a batch.
const NUMBERS: [&str; 8] = ["0", "1", "2", "3", "4", "8", "60", "1000"];

/// Terms of the machine. None may appear in the runner's code or strings. The ROM and boot ROM
/// are named: loading them is the host's job.
const TERMS: [&str; 33] = [
    "gbc", "gameboy", "dmg", "cgb", "cpu", "apu", "ppu", "lcd", "lcdc", "oam", "vram", "wram", "hram", "hdma", "dma", "joypad", "mbc", "rtc", "sm83", "sprite", "tile", "palette", "interrupt", "timer", "cartridge", "scanline", "ly", "stat", "vblank", "hblank", "div", "nintendo", "halt",
];

/// The source with `//` comments removed, string literals kept.
fn code(src: &str) -> String {
    src.lines().map(|l| {
        let mut in_str = false;
        let b = l.as_bytes();
        let mut i = 0;
        while i < b.len() {
            match b[i] {
                b'"' if i == 0 || b[i - 1] != b'\\' => in_str = !in_str,
                b'/' if !in_str && b.get(i + 1) == Some(&b'/') => return l[..i].to_string(),
                _ => {}
            }
            i += 1;
        }
        l.to_string()
    }).collect::<Vec<_>>().join("\n")
}

/// `code` with every string literal emptied: format strings are messages, not values.
fn unquoted(code: &str) -> String {
    let mut out = String::new();
    let mut in_str = false;
    let mut prev = ' ';
    for c in code.chars() {
        if c == '"' && prev != '\\' {
            in_str = !in_str;
            out.push(c);
        } else if !in_str {
            out.push(c);
        }
        prev = c;
    }
    out
}

fn words(code: &str) -> Vec<String> {
    code.split(|c: char| !c.is_ascii_alphanumeric() && c != '_').filter(|w| !w.is_empty()).map(str::to_string).collect()
}

/// The shared transport, up to its tests.
fn transport() -> String {
    let src = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/transport.rs")).unwrap();
    src.split("#[cfg(test)]").next().unwrap().to_string()
}

/// The runner's files and the transport, comments stripped.
fn sources() -> Vec<(String, String)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out: Vec<(String, String)> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| (e.file_name().to_string_lossy().into_owned(), code(&std::fs::read_to_string(e.path()).unwrap()))).collect();
    out.push(("transport.rs".into(), code(&transport())));
    out.sort();
    out
}

/// A line that branches or picks: a condition, a match arm, a loop, a filter, a minimum, an
/// assertion or a panic (a run that stops on a condition decides as much as one that
/// branches on it), or a Boolean `&&` or `||` (which evaluates its right side only on a
/// condition). A closure's empty parameter list `||` is not an operator.
fn branches(code: &str) -> bool {
    let words = words(code);
    words.iter().any(|w| ["if", "else", "match", "while", "loop", "for", "filter", "find", "min", "max", "then", "then_some", "unwrap_or", "unwrap_or_else", "contains", "any", "all", "position", "assert", "assert_eq", "assert_ne", "debug_assert", "debug_assert_eq", "debug_assert_ne", "panic", "unreachable"].contains(&w.as_str())) || code.contains("=>") || code.contains("&&") || boolean_or(code)
}

/// Whether `code` holds `||` as an operator: at the start of a line (a condition continued
/// from the line above) or after an operand (a name, a literal, a closing bracket), not where
/// a closure's parameter list would stand (after an opening bracket, a comma, `=`, or the
/// keywords `move`, `return` and `mut`).
fn boolean_or(code: &str) -> bool {
    let b = code.as_bytes();
    (0..b.len().saturating_sub(1)).any(|i| {
        if &b[i..i + 2] != b"||" {
            return false;
        }
        let before = code[..i].trim_end();
        if before.is_empty() {
            return true;
        }
        let last = before.rsplit(|c: char| !c.is_ascii_alphanumeric() && c != '_').next().unwrap_or("");
        let operand = before.ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == ')' || c == ']' || c == '}' || c == '?');
        operand && !["move", "return", "mut"].contains(&last)
    })
}

/// Each branching line as `file | code`, whitespace collapsed and string literals emptied.
fn branch_lines() -> Vec<String> {
    let mut out = Vec::new();
    for (file, code) in sources() {
        for line in unquoted(&code).lines() {
            let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
            if branches(&line) {
                out.push(format!("{file} | {line}"));
            }
        }
    }
    out
}

/// The review: every branching line of the runner and the transport, each with the class it
/// was given ahead of it (`a`: pure transport, counting down a number a pass emitted,
/// concatenating its bits or moving an output to an input; `b`: presentation; `guard`: a
/// check of what a pass emitted). A line that is not listed, or listed with no class or a
/// class (c), fails: a rule of the machine came back, or a line changed and needs review.
/// `QUARANTINE_LIST=1` rewrites the list, keeping each listed line's class and marking a new
/// one `?`, for a reviewer to classify.
#[test]
fn every_branch_is_reviewed() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/branches.txt");
    let listed = std::fs::read_to_string(&path).unwrap_or_default();
    let class: std::collections::HashMap<String, String> = listed.lines().filter_map(|l| l.split_once(" :: ")).map(|(c, line)| (line.to_string(), c.to_string())).collect();
    let lines = branch_lines();
    if std::env::var("QUARANTINE_LIST").is_ok() {
        let out: Vec<String> = lines.iter().map(|l| format!("{} :: {l}", class.get(l).map_or("?", String::as_str))).collect();
        std::fs::write(&path, out.join("\n") + "\n").unwrap();
    }
    let unreviewed: Vec<&String> = lines.iter().filter(|l| !class.get(*l).is_some_and(|c| ["a", "b", "guard"].iter().any(|k| c.starts_with(k)))).collect();
    assert!(unreviewed.is_empty(), "{} branching lines with no review (tests/branches.txt):\n{}", unreviewed.len(), unreviewed.iter().map(|l| l.as_str()).collect::<Vec<_>>().join("\n"));
    let stale: Vec<&str> = class.keys().map(String::as_str).filter(|l| !lines.iter().any(|x| x == l)).collect();
    assert!(stale.is_empty(), "listed lines no source has now:\n{}", stale.join("\n"));
}

#[test]
fn no_hexadecimal_or_unlisted_number() {
    for (file, code) in sources() {
        for w in words(&unquoted(&code)) {
            let first = w.chars().next().unwrap();
            if !first.is_ascii_digit() {
                continue;
            }
            let bare = ["u8", "u16", "u32", "u64", "usize", "i16", "i32", "i64", "f64"].iter().fold(w.as_str(), |s, t| s.strip_suffix(t).unwrap_or(s));
            assert!(!w.starts_with("0x") && !w.starts_with("0b"), "{file}: the literal {w}");
            assert!(NUMBERS.contains(&bare.replace('_', "").as_str()), "{file}: the literal {w} is not in the list of units");
        }
    }
}

#[test]
fn no_term_of_the_machine() {
    for (file, code) in sources() {
        let lower = code.to_lowercase();
        for w in lower.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty()) {
            for part in w.split('_') {
                assert!(!TERMS.contains(&part), "{file}: the term {part} (in {w})");
            }
        }
        for t in ["game boy", "game_boy"] {
            assert!(!lower.contains(t), "{file}: {t}");
        }
    }
}

/// Every slot name of the model's streams, from the GGUF in `GBCFORMER_GGUF`.
#[test]
fn no_slot_name() {
    let Ok(path) = std::env::var("GBCFORMER_GGUF") else {
        eprintln!("no_slot_name skipped: set GBCFORMER_GGUF to the one model's GGUF, whose slot names it reads");
        return;
    };
    let json = gbcformer::gguf::read_string(Path::new(&path), gbcformer::gguf::META_KEY).unwrap();
    let meta: gbcformer::gguf::Meta = serde_json::from_str(&json).unwrap();
    let names: std::collections::BTreeSet<String> = meta.streams.iter().flat_map(|s| s.slots.iter().cloned()).collect();
    assert!(names.len() > 100, "{} slot names", names.len());
    for (file, code) in sources() {
        let mut rest = code.as_str();
        while let Some(at) = rest.find('"') {
            let after = &rest[at + 1..];
            let end = after.find('"').unwrap_or(after.len());
            let lit = &after[..end];
            assert!(!names.contains(lit), "{file}: the slot name \"{lit}\"");
            rest = &after[(end + 1).min(after.len())..];
        }
    }
}

/// The scanners see a rule written the way one would be: an address, a register's name
/// and a slot name, each outside a comment; and the ratchet sees a branch (the host's
/// "a drop falls at the next cycle") that no review classified.
#[test]
fn the_scanners_catch_a_rule() {
    assert!(branches("self.edges = [cut.drop.then_some(1), cut.edge];"), "a flag turned into a distance branches");
    assert!(branches("while self.more() {"), "a loop on a state bit branches");
    assert!(!branches("self.owed = cut.owed;"), "moving a number does not");
    // The HALT-bug refusal (once in `src/transport.rs`), an assertion the list once missed.
    assert!(branches("assert!(!reread || nibble[DATA] == 0, \"refused\");"), "an assertion decides");
    assert!(branches("debug_assert!(x);"), "a debug assertion decides");
    assert!(branches("let stop = a.done && b.done;"), "a Boolean and decides");
    assert!(branches("let stop = a.done || b.done;"), "a Boolean or decides");
    assert!(!branches("let v = cell.get_or_init(|| make());"), "a closure's empty parameter list does not");
    assert!(!branches("std::thread::spawn(move || work());"), "nor a moved closure's");
    assert!(!branches("s.pass(event, &mut || length);"), "nor a closure borrowed mutably");
    assert!(branches("let x = set.remove || other;"), "a name that ends in a keyword is an operand");
    assert!(branches("|| self.flag"), "a condition continued on its own line joins");
    let src = code("let lcdc = 0xFF40; // the LCD's control\nlet x = \"WQ\";");
    let numbers: Vec<String> = words(&unquoted(&src)).into_iter().filter(|w| w.starts_with(|c: char| c.is_ascii_digit())).collect();
    assert_eq!(numbers, vec!["0xFF40".to_string()]);
    assert!(src.to_lowercase().split(|c: char| !c.is_ascii_alphanumeric()).any(|w| TERMS.contains(&w)), "lcdc is a term");
    assert!(!src.contains("LCD's"), "comments are stripped");
    assert!(src.contains("\"WQ\""), "string literals are kept for the slot check");
}
