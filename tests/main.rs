use pitch_detection::float::Float;
use pitch_detection::utils::buffer::new_real_buffer;
use pitch_detection::detector::PitchDetector;
use pitch_detection::detector::autocorrelation::AutocorrelationDetector;
use pitch_detection::detector::mcleod::McLeodDetector;

#[test]
fn autocorrelation_pure_frequency() {
    pure_frequency(String::from("Autocorrelation"));
}

#[test]
fn mcleod_pure_frequency() {
    pure_frequency(String::from("McLeod"));
}

fn get_chunk<T: Float>(signal: &[T], start: usize, window: usize, output: &mut [T]) {
    let start = match signal.len() > start {
        true => start,
        false => signal.len()
    };

    let stop = match signal.len() >= start + window {
        true => start + window,
        false => signal.len()
    };

    for i in 0..stop - start {
        output[i] = signal[start + i];
    }

    for i in stop - start..output.len() {
        output[i] = T::zero();
    }
}

fn sin_signal<T: Float>(freq: f64, size: usize, sample_rate: usize) -> Vec<T> {
    let mut signal = new_real_buffer(size);
    let two_pi = 2.0 * std::f64::consts::PI;
    let dx = two_pi * freq / sample_rate as f64;
    for i in 0..size {
        let x = i as f64 * dx;
        let y = x.sin();
        signal[i] = T::from(y).unwrap();
    }
    signal
}

fn detector_factory(name: String, window: usize, padding: usize) -> Box<dyn PitchDetector<f64>> {
    match name.as_ref() {
        "McLeod" => {
            return Box::new(McLeodDetector::<f64>::new(window, padding));
        },
        "Autocorrelation" => {
            return Box::new(AutocorrelationDetector::<f64>::new(window, padding));
        },
        _ => {
            panic!("Unknown detector {}", name);
        }
    }
    
}

fn pure_frequency(detector_name: String) {
    const SAMPLE_RATE : usize = 48000;
    const FREQUENCY : f64 = 440.0;
    const DURATION : f64 = 4.0;
    const SAMPLE_SIZE : usize = (SAMPLE_RATE as f64 * DURATION) as usize;
    const WINDOW : usize = 1024;
    const PADDING : usize = WINDOW / 2;
    const DELTA_T : usize = WINDOW / 4;
    const N_WINDOWS : usize = (SAMPLE_SIZE - WINDOW) / DELTA_T;
    const POWER_THRESHOLD : f64 = 500.0;
    const CLARITY_THRESHOLD : f64 = 0.5;

    let signal = sin_signal::<f64>(FREQUENCY, SAMPLE_SIZE, SAMPLE_RATE);

    let mut chunk = new_real_buffer(WINDOW);

    let mut detector = detector_factory(detector_name, WINDOW, PADDING);

    for i in 0..N_WINDOWS {
        let t : usize = i * DELTA_T;
        get_chunk(&signal, t, WINDOW, &mut chunk);

        let pitch = detector.get_pitch(&chunk, SAMPLE_RATE, POWER_THRESHOLD, CLARITY_THRESHOLD);

        match pitch {
            Some(pitch) => {
                let frequency = pitch.frequency;
                let clarity = pitch.clarity;
                let idx = SAMPLE_RATE as f64 / frequency;
                let epsilon = (SAMPLE_RATE as f64 / (idx - 1.0)) - frequency;
                println!("Chosen Peak idx: {}; clarity: {}; freq: {} +/- {}", idx, clarity, frequency, epsilon);
                println!("{}", (frequency - FREQUENCY).abs() < epsilon);
                assert!((frequency - FREQUENCY).abs() < epsilon);
            },
            None => {
                println!("No peaks accepted.");
                assert!(false);
            }
        }
    }
}
