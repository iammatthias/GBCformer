//! The threads that run passes, on the performance cores.
//!
//! At the default QoS, macOS may run a thread on an efficiency core or behind other
//! work at the same class, so a busy machine slows every pass. Each thread that runs
//! passes (the machine's, which runs the GBC CPU, the APU's, the PPU's and their
//! helpers) asks for the user-interactive class, as a game's threads do. It changes
//! only scheduling, never what a pass computes.

/// Asks macOS to keep the calling thread on the performance cores, ahead of default-QoS
/// work. On other platforms it does nothing.
#[cfg(target_os = "macos")]
pub fn interactive() {
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(class: u32, relative_priority: i32) -> i32;
    }
    // SAFETY: sets the calling thread's own QoS class; no pointers are passed.
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

#[cfg(not(target_os = "macos"))]
pub fn interactive() {}
