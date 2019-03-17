# pitch_detection

## Usage
```rust
use pitch_detection::{AutocorrelationDetector, McLeodDetector};

const SAMPLE_RATE : usize = 44100;
const SIZE : usize = 1024;
const PADDING : usize = SIZE / 2;
const CLARITY_THRESHOLD : f64 = 0.7;
const POWER_THRESHOLD : f64 = 5.0;

let signal = vec![0.0; SIZE];
let mut detector = McLeodDetector::new(SIZE, PADDING);

let pitch = detector.get_pitch(&signal, SAMPLE_RATE, POWER_THRESHOLD, CLARITY_THRESHOLD).unwrap();

println!("Frequency: {}, Clarity: {}", pitch.frequency, pitch.clarity);
```
