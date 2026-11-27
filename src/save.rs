//! Battery save files (FR06), native only. A write goes to a sibling temp file, is fsynced, then
//! renamed over the save, so an interrupted write leaves the previous save intact.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

/// `game.gbc` -> `game.sav`.
pub fn path_for(rom: &Path) -> PathBuf {
    rom.with_extension("sav")
}

/// A temp name no other write shares, so two processes (or two saves in flight) saving
/// the same game never write into one temp file.
fn temp_path(path: &Path) -> PathBuf {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let mut name = path.as_os_str().to_owned();
    name.push(format!(".{}.{}.tmp", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    PathBuf::from(name)
}

/// The save bytes, if a save exists. The cartridge checks the size.
pub fn load(path: &Path) -> Result<Option<Vec<u8>>, String> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = temp_path(path);
    let written = File::create(&tmp).and_then(|mut file| {
        file.write_all(bytes)?;
        file.sync_all()
    });
    if let Err(e) = written.and_then(|()| fs::rename(&tmp, path)) {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        File::open(dir)?.sync_all()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("gbcformer-save-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir.join("game.sav")
    }

    #[test]
    fn replaces_save_and_leaves_other_temps_alone() {
        let path = scratch("replace");
        fs::write(&path, [1, 2, 3, 4]).unwrap();
        let other = path.with_extension("sav.1.0.tmp");
        fs::write(&other, [0xEE; 7]).unwrap();
        write_atomic(&path, &[9, 8, 7, 6]).unwrap();
        assert_eq!(fs::read(&path).unwrap(), vec![9, 8, 7, 6]);
        assert_eq!(fs::read(&other).unwrap(), vec![0xEE; 7], "another writer's temp file");
        let left: Vec<_> = fs::read_dir(path.parent().unwrap()).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(left.len(), 2, "no temp file of ours is left: {left:?}");
    }

    #[test]
    fn temp_names_carry_the_pid_and_never_repeat() {
        let path = Path::new("/saves/game.sav");
        let (a, b) = (temp_path(path), temp_path(path));
        let prefix = format!("/saves/game.sav.{}.", std::process::id());
        assert!(a.to_str().unwrap().starts_with(&prefix) && a.to_str().unwrap().ends_with(".tmp"), "{a:?}");
        assert_ne!(a, b);
    }

    #[test]
    fn failed_write_leaves_the_directory_as_it_was() {
        let blocker = scratch("fail");
        fs::write(&blocker, [1, 2, 3, 4]).unwrap();
        // The target's parent is a regular file, so no user, root included, can create
        // the temp file there.
        let path = blocker.join("game.sav");
        let result = write_atomic(&path, &[9, 9, 9, 9]);
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::NotADirectory);
        assert_eq!(fs::read(&blocker).unwrap(), vec![1, 2, 3, 4]);
        let left: Vec<_> = fs::read_dir(blocker.parent().unwrap()).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(left, [std::ffi::OsString::from("game.sav")], "nothing else was created");
    }

    #[test]
    fn load_missing_and_present() {
        let path = scratch("load");
        assert_eq!(load(&path).unwrap(), None);
        fs::write(&path, [5, 6, 7, 8]).unwrap();
        assert_eq!(load(&path).unwrap(), Some(vec![5, 6, 7, 8]));
    }
}
