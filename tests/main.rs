use std::convert::TryInto;
use std::fs;
use std::io::prelude::*;

use pitch_detection::detector::autocorrelation::AutocorrelationDetector;
use pitch_detection::detector::mcleod::McLeodDetector;
use pitch_detection::detector::polyphonic::comb::CombDetector;
use pitch_detection::detector::PitchDetector;
use pitch_detection::detector::PolyphonicDetector;
use pitch_detection::float::Float;
use pitch_detection::utils::buffer::new_real_buffer;

#[test]
fn autocorrelation_sin_signal() {
    mono_frequency("Autocorrelation", "sin", 440.0);
}

#[test]
fn mcleod_sin_signal() {
    mono_frequency("McLeod", "sin", 440.0);
}

#[test]
fn autocorrelation_square_signal() {
    mono_frequency("Autocorrelation", "square", 440.0);
}

#[test]
fn mcleod_square_signal() {
    mono_frequency("McLeod", "square", 440.0);
}

#[test]
fn autocorrelation_triange_signal() {
    mono_frequency("Autocorrelation", "triangle", 440.0);
}

#[test]
fn mcleod_triangle_signal() {
    mono_frequency("McLeod", "triangle", 440.0);
}

#[test]
fn mcleod_poly_square_signal() {
    poly_frequency("McLeod", "square", &[440.0, 523.25, 659.25]);
}

#[test]
fn mcleod_mono_guitar_wave() {
    mono_wave("McLeod", "./tests/guitar_F.bin", 44100, &[349.23], 30., 1.);
    mono_wave("McLeod", "./tests/guitar_A.bin", 44100, &[440.], 30., 1.);
}

#[test]
fn mcleod_mono_mandolin_wave() {
    mono_wave(
        "McLeod",
        "./tests/mandolin_F#.bin",
        44100,
        &[369.99],
        10.,
        1.,
    );
    mono_wave("McLeod", "./tests/mandolin_A.bin", 44100, &[440.], 10., 1.);
}

#[test]
fn mcleod_poly_guitar_wave() {
    mono_wave(
        "McLeod",
        "./tests/guitar_FA.bin",
        44100,
        &[349.23, 440.0],
        100.,
        0.7,
    );
}

#[test]
fn mcleod_poly_mandolin_wave() {
    mono_wave(
        "McLeod",
        "./tests/mandolin_F#A.bin",
        44100,
        &[369.99, 440.0],
        30.,
        1.,
    );
}

fn get_chunk<T: Float>(signal: &[T], start: usize, window: usize, output: &mut [T]) {
    let start = match signal.len() > start {
        true => start,
        false => signal.len(),
    };

    let stop = match signal.len() >= start + window {
        true => start + window,
        false => signal.len(),
    };

    for i in 0..stop - start {
        output[i] = signal[start + i];
    }

    for i in stop - start..output.len() {
        output[i] = T::zero();
    }
}

fn sin_wave<T: Float>(freq: f64, size: usize, sample_rate: usize) -> Vec<T> {
    let mut signal = new_real_buffer(size, T::zero());
    let two_pi = 2.0 * std::f64::consts::PI;
    let dx = two_pi * freq / sample_rate as f64;
    for i in 0..size {
        let x = i as f64 * dx;
        let y = x.sin();
        signal[i] = T::from(y).unwrap();
    }
    signal
}

fn square_wave<T: Float>(freq: f64, size: usize, sample_rate: usize) -> Vec<T> {
    let mut signal = new_real_buffer(size, T::zero());
    let period = sample_rate as f64 / freq;

    for i in 0..size {
        let x = i as f64 / period;
        let frac = x - x.floor();
        let y = match frac >= 0.5 {
            true => -1.0,
            false => 1.0,
        };
        signal[i] = T::from(y).unwrap();
    }
    signal
}

fn triangle_wave<T: Float>(freq: f64, size: usize, sample_rate: usize) -> Vec<T> {
    let mut signal = new_real_buffer(size, T::zero());
    let period = sample_rate as f64 / freq;

    for i in 0..size {
        let x = i as f64 / period;
        let frac = x - x.floor();
        let y = match frac {
            f if f >= 0. && f < 0.25 => 4. * f,
            f if f >= 0.25 && f < 0.75 => 1. - 4. * (f - 0.25),
            f if f >= 0.75 && f < 1. => -1. + 4. * (f - 0.75),
            _ => panic!("Should be between 0 and 1"),
        };
        signal[i] = T::from(y).unwrap();
    }
    signal
}

fn saw_wave<T: Float>(freq: f64, size: usize, sample_rate: usize) -> Vec<T> {
    let mut signal = new_real_buffer(size, T::zero());
    let period = sample_rate as f64 / freq;

    for i in 0..size {
        let x = i as f64 / period;
        let frac = x - x.floor();
        let y = match frac {
            f if f >= 0. && f < 0.25 => 4. * f,
            f if f >= 0.25 && f < 0.75 => -1. + 4. * (f - 0.25),
            f if f >= 0.75 && f < 1. => -1. + 4. * (f - 0.75),
            _ => panic!("Should be between 0 and 1"),
        };
        signal[i] = T::from(y).unwrap();
    }
    signal
}

fn detector_factory<T: Float + std::iter::Sum>(
    name: &str,
    window: usize,
    padding: usize,
) -> Box<dyn PitchDetector<T>> {
    match name.as_ref() {
        "McLeod" => {
            return Box::new(McLeodDetector::<T>::new(window, padding));
        }
        "Autocorrelation" => {
            return Box::new(AutocorrelationDetector::<T>::new(window, padding));
        }
        _ => {
            panic!("Unknown detector {}", name);
        }
    }
}

fn signal_factory<T: Float>(name: &str, freq: f64, size: usize, sample_rate: usize) -> Vec<T> {
    match name.as_ref() {
        "sin" => {
            return sin_wave(freq, size, sample_rate);
        }
        "square" => {
            return square_wave(freq, size, sample_rate);
        }
        "triangle" => {
            return triangle_wave(freq, size, sample_rate);
        }
        "saw" => {
            return saw_wave(freq, size, sample_rate);
        }
        _ => {
            panic!("Unknown wave function {}", name);
        }
    }
}

fn is_same_note_any_octave<T: Float>(a: T, b: T, cents_threshold: T) -> bool {
    let one = T::one();
    let two = one + one;

    let (lower_freq, higher_freq) = match a < b {
        true => (a, b),
        false => (b, a),
    };

    let mut best_ratio = higher_freq / lower_freq;

    let mut scaled_lower_freq = lower_freq * two;
    let mut curr_ratio = higher_freq / scaled_lower_freq;
    while (curr_ratio - one).abs() < (best_ratio - one).abs() {
        best_ratio = curr_ratio;
        scaled_lower_freq = scaled_lower_freq * two;
        curr_ratio = higher_freq / scaled_lower_freq;
    }

    let cents = T::from(1200.0).unwrap() * best_ratio.ln() / two.ln();

    cents.abs() < cents_threshold
}

fn mono_frequency(detector_name: &str, wave_name: &str, freq_in: f64) {
    const SAMPLE_RATE: usize = 48000;
    const DURATION: f64 = 4.0;
    const SAMPLE_SIZE: usize = (SAMPLE_RATE as f64 * DURATION) as usize;
    const WINDOW: usize = 1024;
    const PADDING: usize = WINDOW / 2;
    const DELTA_T: usize = WINDOW / 4;
    const N_WINDOWS: usize = (SAMPLE_SIZE - WINDOW) / DELTA_T;
    const POWER_THRESHOLD: f64 = 300.0;
    const CLARITY_THRESHOLD: f64 = 0.6;

    let signal = signal_factory::<f64>(wave_name, freq_in, SAMPLE_SIZE, SAMPLE_RATE);

    let mut chunk = new_real_buffer(WINDOW, 0.0_f64);

    let mut detector = detector_factory(detector_name, WINDOW, PADDING);

    for i in 0..N_WINDOWS {
        let t: usize = i * DELTA_T;
        get_chunk(&signal, t, WINDOW, &mut chunk);

        let pitch = detector.get_pitch(&chunk, SAMPLE_RATE, POWER_THRESHOLD, CLARITY_THRESHOLD);

        match pitch {
            Some(pitch) => {
                let frequency = pitch.frequency;
                let clarity = pitch.clarity;
                let idx = SAMPLE_RATE as f64 / frequency;
                let epsilon = (SAMPLE_RATE as f64 / (idx - 1.0)) - frequency;
                println!(
                    "Chosen Peak idx: {}; clarity: {}; freq: {} +/- {}",
                    idx, clarity, frequency, epsilon
                );
                assert!((frequency - freq_in).abs() < 2. * epsilon);
            }
            None => {
                println!("No peaks accepted.");
                assert!(false);
            }
        }
    }
}

fn poly_frequency(detector_name: &str, wave_name: &str, freqs_in: &[f64]) {
    const SAMPLE_RATE: usize = 48000;
    const DURATION: f64 = 4.0;
    const SAMPLE_SIZE: usize = (SAMPLE_RATE as f64 * DURATION) as usize;
    const WINDOW: usize = 1024 * 4;
    const PADDING: usize = WINDOW / 2;
    const DELTA_T: usize = WINDOW / 4;
    const N_WINDOWS: usize = (SAMPLE_SIZE - WINDOW) / DELTA_T;
    const POWER_THRESHOLD: f64 = 300.0;
    const CLARITY_THRESHOLD: f64 = 0.6;

    let mut signal = new_real_buffer(SAMPLE_SIZE, 0.0_f64);

    for freq in freqs_in.iter() {
        let freq_signal = signal_factory::<f64>(wave_name, *freq, SAMPLE_SIZE, SAMPLE_RATE);
        for (s0, s1) in freq_signal.iter().zip(signal.iter_mut()) {
            *s1 = *s1 + *s0;
        }
    }

    let mut chunk = new_real_buffer(WINDOW, 0.0_f64);

    let monophonic_detector = detector_factory(detector_name, WINDOW, PADDING);
    let mut detector = Box::new(CombDetector::<f64>::new(
        monophonic_detector,
        freqs_in.len(),
    ));

    let n_tests = N_WINDOWS * freqs_in.len();
    let mut failures = 0;

    for i in 0..N_WINDOWS {
        let t: usize = i * DELTA_T;
        get_chunk(&signal, t, WINDOW, &mut chunk);

        let pitches = detector.get_pitch(&chunk, SAMPLE_RATE, POWER_THRESHOLD, CLARITY_THRESHOLD);

        let mut remaining_freqs_in = vec![0.0; freqs_in.len()];
        remaining_freqs_in.copy_from_slice(freqs_in);

        for (pitch_i, pitch) in pitches.iter().enumerate() {
            match pitch {
                Some(pitch) => {
                    let frequency = pitch.frequency;
                    let clarity = pitch.clarity;
                    let idx = SAMPLE_RATE as f64 / frequency;
                    let epsilon = (SAMPLE_RATE as f64 / (idx - 1.0)) - frequency;
                    println!(
                        "Poly #{} - Chosen Peak idx: {}; clarity: {}; freq: {} +/- {}",
                        pitch_i, idx, clarity, frequency, epsilon
                    );

                    let mut n_accepted = 0;
                    let mut accepted = vec![false; remaining_freqs_in.len()];
                    for (index, freq_in) in remaining_freqs_in.iter().enumerate() {
                        // Not very stable, accept wrong octaves too...
                        if is_same_note_any_octave(frequency, *freq_in, 20.) {
                            accepted[index] = true;
                            n_accepted += 1;
                        }
                    }

                    if n_accepted != 1 {
                        failures += 1;
                    }

                    let mut index = 0;
                    remaining_freqs_in.retain(|_| {
                        let keep = !accepted[index];
                        index += 1;
                        keep
                    });
                }
                None => {
                    println!("Poly #{} - No peaks accepted.", pitch_i);
                    failures += 1;
                }
            }
        }
    }

    let frac_failures = (failures as f64) / (n_tests as f64);
    println!(
        "FAILURES: {} TOTAL: {}  FRAC: {}",
        failures, n_tests, frac_failures
    );

    assert!(frac_failures < 0.2);
}

fn mono_wave(
    detector_name: &str,
    file_name: &str,
    sample_rate: usize,
    freqs_in: &[f32],
    cents_threshold: f32,
    success_rate: f32,
) {
    const WINDOW: usize = 1024 * 4;
    const PADDING: usize = WINDOW / 2;
    const DELTA_T: usize = WINDOW / 4;
    const POWER_THRESHOLD: f32 = 5.0;
    const CLARITY_THRESHOLD: f32 = 0.6;

    // let N_WINDOWS: usize = (SAMPLE_SIZE - WINDOW) / DELTA_T;
    let monophonic_detector = detector_factory(detector_name, WINDOW, PADDING);
    let mut detector = Box::new(CombDetector::<f32>::new(
        monophonic_detector,
        freqs_in.len(),
    ));

    let mut file = fs::File::open(file_name).unwrap();
    let mut signal = Vec::<f32>::new();
    let mut buffer = Vec::<u8>::new();
    let buffer_size = file.read_to_end(&mut buffer).unwrap();

    let signal_size = buffer_size / 4;
    for i in 0..signal_size {
        let b: [u8; 4] = (&buffer[i * 4..i * 4 + 4]).try_into().unwrap();
        signal.push(f32::from_le_bytes(b));
    }

    let N_WINDOWS: usize = (signal_size - WINDOW) / DELTA_T;

    let mut chunk = new_real_buffer(WINDOW, 0.0_f32);

    let n_tests = N_WINDOWS * freqs_in.len();
    let mut failures = 0;

    for i in 0..N_WINDOWS {
        let t: usize = i * DELTA_T;
        println!("Window t = {} s", t as f32 / sample_rate as f32);
        get_chunk(&signal, t, WINDOW, &mut chunk);

        let pitches = detector.get_pitch(&chunk, sample_rate, POWER_THRESHOLD, CLARITY_THRESHOLD);

        let mut remaining_freqs_in = vec![0.0; freqs_in.len()];
        remaining_freqs_in.copy_from_slice(freqs_in);

        for (pitch_i, pitch) in pitches.iter().enumerate() {
            match pitch {
                Some(pitch) => {
                    let frequency = pitch.frequency;
                    let clarity = pitch.clarity;
                    let idx = sample_rate as f32 / frequency;
                    let epsilon = (sample_rate as f32 / (idx - 1.0)) - frequency;
                    println!(
                        "Poly #{} - Chosen Peak idx: {}; clarity: {}; freq: {} +/- {}",
                        pitch_i, idx, clarity, frequency, epsilon
                    );

                    let mut n_accepted = 0;
                    let mut accepted = vec![false; remaining_freqs_in.len()];
                    for (index, freq_in) in remaining_freqs_in.iter().enumerate() {
                        // Not very stable, accept wrong octaves too...
                        if is_same_note_any_octave(frequency, *freq_in, cents_threshold) {
                            accepted[index] = true;
                            n_accepted += 1;
                        }
                    }

                    if n_accepted != 1 {
                        failures += 1;
                    }

                    let mut index = 0;
                    remaining_freqs_in.retain(|_| {
                        let keep = !accepted[index];
                        index += 1;
                        keep
                    });
                }
                None => {
                    println!("Poly #{} - No peaks accepted.", pitch_i);
                    failures += 1;
                }
            }
        }
    }

    let frac_failures = (failures as f32) / (n_tests as f32);
    println!(
        "FAILURES: {} TOTAL: {}  FRAC: {}",
        failures, n_tests, frac_failures
    );

    assert!(frac_failures <= 1.0 - success_rate);
}
