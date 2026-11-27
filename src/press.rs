//! Scripted button presses, written `FRAME:BUTTON[:HOLD]`.

use crate::machine::io::Buttons;

/// Frames a press is held when its spec names no hold.
pub const DEFAULT_HOLD: u64 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Press {
    pub frame: u64,
    pub buttons: u8,
    pub hold: u64,
}

impl Press {
    pub fn parse(spec: &str) -> Result<Press, String> {
        let bad = || format!("{spec}: expected FRAME:BUTTON[:HOLD]");
        let parts: Vec<&str> = spec.split(':').collect();
        let (frame, name, hold) = match parts[..] {
            [frame, name] => (frame, name, None),
            [frame, name, hold] => (frame, name, Some(hold)),
            _ => return Err(bad()),
        };
        let frame = frame.parse().map_err(|_| bad())?;
        let buttons = Buttons::parse(name)
            .ok_or_else(|| format!("{spec}: unknown button {name:?}; buttons are up down left right a b start select"))?;
        let hold = hold.map_or(Ok(DEFAULT_HOLD), |h| h.parse().map_err(|_| bad()))?;
        if hold == 0 {
            return Err(format!("{spec}: HOLD must be at least 1"));
        }
        Ok(Press { frame, buttons, hold })
    }

    fn held_at(&self, frame: u64) -> bool {
        frame >= self.frame && frame - self.frame < self.hold
    }
}

/// Presses that may overlap; overlapping presses hold all their buttons.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PressScript(Vec<Press>);

impl PressScript {
    pub fn parse<S: AsRef<str>>(specs: &[S]) -> Result<PressScript, String> {
        specs.iter().map(|s| Press::parse(s.as_ref())).collect::<Result<_, _>>().map(PressScript)
    }

    pub fn push(&mut self, press: Press) {
        self.0.push(press);
    }

    pub fn buttons_at(&self, frame: u64) -> Buttons {
        Buttons(self.0.iter().filter(|p| p.held_at(frame)).fold(0, |b, p| b | p.buttons))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_specs_with_and_without_hold() {
        assert_eq!(Press::parse("120:Start"), Ok(Press { frame: 120, buttons: Buttons::START, hold: 3 }));
        assert_eq!(Press::parse("7:left:24"), Ok(Press { frame: 7, buttons: Buttons::LEFT, hold: 24 }));
    }

    #[test]
    fn rejects_unknown_buttons_and_bad_shapes() {
        assert_eq!(Press::parse("5:jump"), Err("5:jump: unknown button \"jump\"; buttons are up down left right a b start select".into()));
        assert_eq!(Press::parse("5"), Err("5: expected FRAME:BUTTON[:HOLD]".into()));
        assert_eq!(Press::parse("x:a"), Err("x:a: expected FRAME:BUTTON[:HOLD]".into()));
        assert_eq!(Press::parse("5:a:2:9"), Err("5:a:2:9: expected FRAME:BUTTON[:HOLD]".into()));
        assert!(PressScript::parse(&["1:a", "2:selct"]).is_err());
    }

    #[test]
    fn rejects_a_hold_of_zero_frames() {
        assert_eq!(Press::parse("5:a:0"), Err("5:a:0: HOLD must be at least 1".into()));
        assert_eq!(Press::parse("5:a:1"), Ok(Press { frame: 5, buttons: Buttons::A, hold: 1 }));
    }

    #[test]
    fn a_press_at_the_last_frame_is_held_there() {
        let script = PressScript::parse(&["18446744073709551615:start"]).unwrap();
        assert_eq!(script.buttons_at(u64::MAX).0, Buttons::START);
        assert_eq!(script.buttons_at(u64::MAX - 1).0, 0);
    }

    #[test]
    fn overlapping_presses_combine_for_their_hold() {
        let script = PressScript::parse(&["10:a:4", "12:down"]).unwrap();
        let held: Vec<u8> = (9..16).map(|f| script.buttons_at(f).0).collect();
        let (a, down) = (Buttons::A, Buttons::DOWN);
        assert_eq!(held, [0, a, a, a | down, a | down, down, 0]);
    }
}
