//! The calling thread's own counters: CPU time, and on Apple silicon the instructions
//! retired and the cycles run. They count only while the thread runs, so a busy machine
//! that takes the core away moves wall time but not these.

/// One reading of the calling thread's counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub cpu_ns: u64,
    pub instructions: u64,
    pub cycles: u64,
}

impl Counts {
    pub fn plus(self, other: Counts) -> Counts {
        Counts { cpu_ns: self.cpu_ns + other.cpu_ns, instructions: self.instructions + other.instructions, cycles: self.cycles + other.cycles }
    }

    pub fn since(self, earlier: Counts) -> Counts {
        Counts { cpu_ns: self.cpu_ns - earlier.cpu_ns, instructions: self.instructions - earlier.instructions, cycles: self.cycles - earlier.cycles }
    }
}

/// The calling thread's counters, or None where the platform gives none. Instructions
/// and cycles are 0 where the kernel does not count them per thread.
#[cfg(target_os = "macos")]
pub fn thread() -> Option<Counts> {
    unsafe extern "C" {
        fn clock_gettime_nsec_np(clock: u32) -> u64;
        fn syscall(number: i32, ...) -> i32;
    }
    const CLOCK_THREAD_CPUTIME_ID: u32 = 16;
    // `thread_selfcounts(THSC_CPI, buf, 16)`: the thread's instructions, then cycles.
    const SYS_THREAD_SELFCOUNTS: i32 = 186;
    const THSC_CPI: i32 = 1;
    let mut cpi = [0u64; 2];
    // SAFETY: the clock call takes a constant; the syscall writes at most 16 bytes to `cpi`.
    let (cpu_ns, ok) = unsafe { (clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID), syscall(SYS_THREAD_SELFCOUNTS, THSC_CPI, cpi.as_mut_ptr(), std::mem::size_of_val(&cpi)) == 0) };
    if !ok {
        cpi = [0; 2];
    }
    Some(Counts { cpu_ns, instructions: cpi[0], cycles: cpi[1] })
}

#[cfg(not(target_os = "macos"))]
pub fn thread() -> Option<Counts> {
    None
}

/// The calling thread's instructions and cycles alone (CPU time 0), in one system call:
/// what a bracket around a short span reads.
#[cfg(target_os = "macos")]
pub fn cpi() -> Option<Counts> {
    unsafe extern "C" {
        fn syscall(number: i32, ...) -> i32;
    }
    let mut cpi = [0u64; 2];
    // SAFETY: `thread_selfcounts(THSC_CPI, buf, 16)` writes at most 16 bytes to `cpi`.
    let ok = unsafe { syscall(186, 1, cpi.as_mut_ptr(), std::mem::size_of_val(&cpi)) == 0 };
    ok.then_some(Counts { cpu_ns: 0, instructions: cpi[0], cycles: cpi[1] })
}

#[cfg(not(target_os = "macos"))]
pub fn cpi() -> Option<Counts> {
    None
}
