use std::f64::consts::PI;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pitch_detection::{
    detector::{autocorrelation::AutocorrelationDetector, mcleod::McLeodDetector, PitchDetector},
    utils::peak::detect_peaks,
};

pub fn utils_benchmark(c: &mut Criterion) {
    let v = (0..1024)
        .into_iter()
        .map(|v| ((v as f64) / PI / 30.).sin())
        .collect::<Vec<f64>>();
    let vv = v.as_slice();

    c.bench_function("detect_peaks", |b| {
        b.iter(|| detect_peaks(black_box(vv)).collect::<Vec<_>>())
    });
}

pub fn pitch_detect_benchmark(c: &mut Criterion) {
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

    let mut mcleod_detector = McLeodDetector::new(SIZE, PADDING);
    let mut autocorrelation_detector = AutocorrelationDetector::new(SIZE, PADDING);

    c.bench_function("McLeod get_pitch", |b| {
        b.iter(|| {
            mcleod_detector
                .get_pitch(
                    black_box(&signal),
                    SAMPLE_RATE,
                    POWER_THRESHOLD,
                    CLARITY_THRESHOLD,
                )
                .unwrap()
        });
    });

    c.bench_function("Autocorrelation get_pitch", |b| {
        b.iter(|| {
            autocorrelation_detector
                .get_pitch(
                    black_box(&signal),
                    SAMPLE_RATE,
                    POWER_THRESHOLD,
                    CLARITY_THRESHOLD,
                )
                .unwrap()
        });
    });
}

criterion_group!(benches, pitch_detect_benchmark, utils_benchmark);
criterion_main!(benches);
