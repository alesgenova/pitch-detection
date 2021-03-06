use std::f64::consts::PI;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pitch_detection::utils::peak::detect_peaks;

pub fn criterion_benchmark(c: &mut Criterion) {
    let v = (0..1024)
        .into_iter()
        .map(|v| ((v as f64) / PI / 30.).sin())
        .collect::<Vec<f64>>();
    let vv = v.as_slice();

    c.bench_function("detect_peaks", |b| {
        b.iter(|| detect_peaks(black_box(vv)).collect::<Vec<_>>())
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
