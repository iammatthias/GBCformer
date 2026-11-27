//! The desktop player: a window, speaker output and frame pacing.

use crate::machine::apu::{SAMPLE_RATE, Sample};
use crate::machine::io::Buttons;
use crate::machine::ppu::{HEIGHT, WIDTH};
use crate::output::Recorder;
use crate::session::{Event, Session};
use crate::step::Core;
use minifb::{Key, Scale, Window, WindowOptions};
use std::time::{Duration, Instant};

/// Frames per second of the DMG and CGB at normal speed.
pub const FPS: f64 = 59.7275;

/// The host clock in Unix seconds, for [`Session::set_time`]; 0 before the epoch.
pub fn unix_seconds() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Plays samples on the default output device. The emulator is the clock, so `send`
/// never blocks, it drops samples that overflow four frames of queue, and the device
/// plays silence when the queue runs dry.
pub struct Speaker {
    queue: std::sync::mpsc::SyncSender<Sample>,
    _stream: cpal::Stream,
}

impl Speaker {
    pub fn open() -> Result<Self, String> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        let device = cpal::default_host().default_output_device().ok_or("no audio output device")?;
        let config = cpal::StreamConfig { channels: 2, sample_rate: SAMPLE_RATE, buffer_size: cpal::BufferSize::Default };
        let frame = SAMPLE_RATE as f64 / FPS;
        let (queue, rx) = std::sync::mpsc::sync_channel::<Sample>((4.0 * frame) as usize);
        // Two frames of lead, so the device does not run dry between frame bursts.
        for _ in 0..(2.0 * frame) as usize {
            let _ = queue.try_send([0, 0]);
        }
        let fill = move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
            for frame in out.chunks_exact_mut(2) {
                let [l, r] = rx.try_recv().unwrap_or([0, 0]);
                frame[0] = l as f32 / 32768.0;
                frame[1] = r as f32 / 32768.0;
            }
        };
        let stream = device
            .build_output_stream(config, fill, |e| eprintln!("audio: {e}"), None)
            .map_err(|e| format!("48 kHz stereo output: {e}"))?;
        stream.play().map_err(|e| e.to_string())?;
        Ok(Speaker { queue, _stream: stream })
    }

    pub fn send(&self, samples: &mut Vec<Sample>) {
        for s in samples.drain(..) {
            let _ = self.queue.try_send(s);
        }
    }
}

/// Holds a loop to the console's frame rate. When a frame runs late it drops the debt
/// instead of running frames back to back to catch up, so a stall (a slow model pass)
/// never turns into a burst of fast frames.
pub struct Pacer {
    frame_time: Duration,
    deadline: Instant,
}

impl Default for Pacer {
    fn default() -> Self {
        Pacer { frame_time: Duration::from_secs_f64(1.0 / FPS), deadline: Instant::now() }
    }
}

impl Pacer {
    /// Sleeps until the frame's time is up. Call it once per frame.
    pub fn wait(&mut self) {
        self.deadline += self.frame_time;
        match self.deadline.checked_duration_since(Instant::now()) {
            Some(wait) => std::thread::sleep(wait),
            None => self.deadline = Instant::now(),
        }
    }
}

pub struct PlayOptions {
    /// Close the window after this many frames.
    pub frames: Option<u64>,
    pub mute: bool,
    pub record: Option<Recorder>,
}

const KEYS: [(Key, u8); 8] = [
    (Key::Right, Buttons::RIGHT),
    (Key::Left, Buttons::LEFT),
    (Key::Up, Buttons::UP),
    (Key::Down, Buttons::DOWN),
    (Key::Z, Buttons::A),
    (Key::X, Buttons::B),
    (Key::Backspace, Buttons::SELECT),
    (Key::Enter, Buttons::START),
];

/// Plays the session in a window at console speed until the window closes, Escape is
/// pressed or `options.frames` have run, with the keyboard as the buttons. Events go to
/// `on_event`.
pub fn play<C: Core>(
    session: &mut Session<C>,
    options: &PlayOptions,
    mut on_event: impl FnMut(Event),
) -> Result<(), String> {
    // A library caller may run the session on its own thread, so this keeps
    // that thread on the performance cores, as `gbcformer` does from `main`.
    crate::qos::interactive();
    let speaker = if options.mute {
        None
    } else {
        Speaker::open().inspect_err(|e| eprintln!("audio off: {e}")).ok()
    };
    let mut audio = Vec::new();
    let window_options = WindowOptions { scale: Scale::X4, ..WindowOptions::default() };
    let mut window = Window::new("gbcformer", WIDTH, HEIGHT, window_options).map_err(|e| e.to_string())?;
    let mut pacer = Pacer::default();
    let (mut window_start, mut window_frames, mut window_counters) = (Instant::now(), 0u32, session.counters());
    while window.is_open() && !window.is_key_down(Key::Escape) && options.frames.is_none_or(|n| session.frame_number() < n) {
        let frame = session.frame_number();
        let keys = Buttons(KEYS.iter().filter(|(k, _)| window.is_key_down(*k)).fold(0, |b, (_, bit)| b | bit));
        session.set_time(unix_seconds());
        for event in session.run_frame(keys) {
            on_event(event);
        }
        session.drain_audio(&mut audio);
        match &speaker {
            Some(speaker) => speaker.send(&mut audio),
            None => audio.clear(),
        }
        if let Some(r) = &options.record {
            r.record(frame, session.frame());
        }
        window.update_with_buffer(&session.frame()[..], WIDTH, HEIGHT).map_err(|e| e.to_string())?;
        window_frames += 1;
        let now = Instant::now();
        if now - window_start >= Duration::from_secs(1) {
            let secs = (now - window_start).as_secs_f64();
            let (c, p) = (session.counters(), window_counters);
            let title = format!(
                "gbcformer  {:.1} fps  steps/s {}  halt M-cycles/s {}",
                window_frames as f64 / secs,
                ((c.steps - p.steps) as f64 / secs) as u64,
                ((c.halt_mcycles - p.halt_mcycles) as f64 / secs) as u64
            );
            window.set_title(&title);
            println!("{title}");
            (window_start, window_frames, window_counters) = (now, 0, c);
        }
        pacer.wait();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pacer_drops_debt_after_a_stall_instead_of_bursting() {
        let mut pacer = Pacer::default();
        std::thread::sleep(pacer.frame_time * 10);
        let start = Instant::now();
        pacer.wait();
        assert!(start.elapsed() < pacer.frame_time, "the late frame does not wait");
        for _ in 0..3 {
            pacer.wait();
        }
        assert!(start.elapsed() >= pacer.frame_time * 3, "frames after the stall run at frame rate, {:?}", start.elapsed());
    }
}
