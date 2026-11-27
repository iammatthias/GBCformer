//! Named slots of a checkpoint's token sequence. `slots!` turns a variant list into a
//! `#[repr(usize)]` enum whose discriminants are the slot indices, with the display
//! names the checkpoint and its debug output use. A variant whose name is not an
//! identifier, such as `A'`, spells it with `= "A'"`.

macro_rules! slots {
    ($(#[$meta:meta])* $vis:vis enum $name:ident { $($variant:ident $(= $label:literal)?),* $(,)? }) => {
        $(#[$meta])*
        #[repr(usize)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        $vis enum $name { $($variant),* }

        impl $name {
            pub const ALL: [Self; { [$(stringify!($variant)),*].len() }] = [$(Self::$variant),*];
            pub const NAMES: [&'static str; { [$(stringify!($variant)),*].len() }] = [$(slots!(@name $variant $(= $label)?)),*];
        }
    };
    (@name $variant:ident = $label:literal) => { $label };
    (@name $variant:ident) => { stringify!($variant) };
}

#[cfg(test)]
mod tests {
    slots! { enum Demo { X, Y = "Y'", Z } }

    #[test]
    fn indices_match_names() {
        assert_eq!(Demo::NAMES, ["X", "Y'", "Z"]);
        assert_eq!(Demo::ALL.map(|v| v as usize), [0, 1, 2]);
        assert_eq!(Demo::NAMES[Demo::Y as usize], "Y'");
    }
}
