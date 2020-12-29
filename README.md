[![workflow status](https://github.com/alesgenova/pitch-detection/workflows/main/badge.svg?branch=master)](https://github.com/alesgenova/pitch-detection/actions?query=workflow%3Amain+branch%3Amaster)
[![crates.io](https://img.shields.io/crates/v/pitch-detection.svg)](https://crates.io/crates/pitch-detection)

# pitch_detection

## Usage
```rust
use pitch_detection::detector::mcleod::McLeodDetector;
use pitch_detection::detector::PitchDetector;

fn main() {
    const SAMPLE_RATE: usize = 44100;
    const SIZE: usize = 1024;
    const PADDING: usize = SIZE / 2;
    const POWER_THRESHOLD: f64 = 5.0;
    const CLARITY_THRESHOLD: f64 = 0.7;

    // Signal coming from some source (microphone, generated, etc...)
    let dt = 1.0 / SAMPLE_RATE as f64;
    let freq = 300.0;
    let signal: Vec<f64> = (0..SIZE)
        .map(|x| (2.0 * std::f64::consts::PI * x as f64 * dt * freq).sin())
        .collect();

    let mut detector = McLeodDetector::new(SIZE, PADDING);

    let pitch = detector
        .get_pitch(&signal, SAMPLE_RATE, POWER_THRESHOLD, CLARITY_THRESHOLD)
        .unwrap();

    println!("Frequency: {}, Clarity: {}", pitch.frequency, pitch.clarity);
}
```
## Live Demo
[![Demo Page](https://raw.githubusercontent.com/alesgenova/pitch-detection-app/master/demo.png)](https://alesgenova.github.io/pitch-detection-app/)
[Source](https://github.com/alesgenova/pitch-detection-app)
