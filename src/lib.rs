//! # Pitch Detection
//! *pitch_detection* implements several algorithms for estimating the
//! fundamental frequency of a sound wave stored in a buffer. It is designed
//! to be usable in a WASM environment.
//!
//! # Detectors
//! A *detector* is an implementation of a pitch detection algorithm. Each detector's tolerance
//! for noise and polyphonic sounds varies.
//!
//!   * [AutocorrelationDetector][detector::autocorrelation]
//!   * [McLeodDetector][detector::mcleod]
//!   * [YINDetector][detector::yin]
//!
//! # Examples
//! ```
//! use pitch_detection::detector::mcleod::McLeodDetector;
//! use pitch_detection::detector::PitchDetector;
//!
//! fn main() {
//!     const SAMPLE_RATE: usize = 44100;
//!     const SIZE: usize = 1024;
//!     const PADDING: usize = SIZE / 2;
//!     const POWER_THRESHOLD: f64 = 5.0;
//!     const CLARITY_THRESHOLD: f64 = 0.7;
//!
//!     // Signal coming from some source (microphone, generated, etc...)
//!     let dt = 1.0 / SAMPLE_RATE as f64;
//!     let freq = 300.0;
//!     let signal: Vec<f64> = (0..SIZE)
//!         .map(|x| (2.0 * std::f64::consts::PI * x as f64 * dt * freq).sin())
//!         .collect();
//!
//!     let mut detector = McLeodDetector::new(SIZE, PADDING);
//!
//!     let pitch = detector
//!         .get_pitch(&signal, SAMPLE_RATE, POWER_THRESHOLD, CLARITY_THRESHOLD)
//!         .unwrap();
//!
//!     println!("Frequency: {}, Clarity: {}", pitch.frequency, pitch.clarity);
//! }
//! ```

pub use detector::internals::Pitch;

pub mod detector;
pub mod float;
pub mod utils;
