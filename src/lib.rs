use num_traits::float::{FloatCore as NumFloatCore};
use num_traits::Zero;
use num_complex::Complex;
use rustfft::FFTplanner;
use rustfft::FFTnum;

pub trait FloatCore : NumFloatCore + FFTnum {}

impl FloatCore for f64 {}
impl FloatCore for f32 {}

enum ComplexComponent {
    Re,
    Im
}

enum PeakCorrection {
    Quadratic,
    None
}

struct Point<T: FloatCore> {
    x: T,
    y: T
}

pub trait PitchDetector<T>
    where T : FloatCore
{
    fn get_pitch(&mut self, signal: &[T], sample_rate: usize, threshold: T) -> Option<Pitch<T>>;
}

pub struct Pitch<T>
    where T: FloatCore
{
    pub frequency: T,
    pub clarity: T
}

struct DetectorInternals<T>
    where T: FloatCore
{
    size: usize,
    padding: usize,
    real_buffers: Vec<Vec<T>>,
    complex_buffers: Vec<Vec<Complex<T>>>
}

impl<T> DetectorInternals<T>
    where T : FloatCore
{
    pub fn new(n_real_buffers: usize, n_complex_buffers: usize, size: usize, padding: usize) -> Self {
        let mut real_buffers: Vec<Vec<T>> = Vec::new();
        let mut complex_buffers: Vec<Vec<Complex<T>>> = Vec::new();

        for _i in 0..n_real_buffers {
            let v = new_real_buffer(size + padding);
            real_buffers.push(v);
        }

        for _i in 0..n_complex_buffers {
            let v = new_complex_buffer(size + padding);
            complex_buffers.push(v);
        }

        DetectorInternals {
            size,
            padding,
            real_buffers,
            complex_buffers
        }
    }
}

pub struct AutocorrelationDetector<T>
    where T : FloatCore
{
    internals: DetectorInternals<T>
}

impl<T> AutocorrelationDetector<T>
    where T: FloatCore
{
    pub fn new(size: usize, padding: usize) -> Self {
        let internals = DetectorInternals::new(1, 2, size, padding);
        AutocorrelationDetector {
            internals
        }
    }
}

impl<T> PitchDetector<T> for AutocorrelationDetector<T>
    where T : FloatCore
{
    fn get_pitch(&mut self, signal: &[T], sample_rate: usize, threshold: T) -> Option<Pitch<T>> {
        assert_eq!(signal.len(), self.internals.size);

        let (signal_complex, rest) = self.internals.complex_buffers.split_first_mut().unwrap();
        let (scratch, _) = rest.split_first_mut().unwrap();
        let (autocorr, _) = self.internals.real_buffers.split_first_mut().unwrap();

        copy_real_to_complex(signal, signal_complex, ComplexComponent::Re);
        autocorrelation(signal_complex, scratch, autocorr);

        let threshold = threshold * autocorr[0];
        let peaks = detect_peaks(autocorr);
        let chosen_peak = choose_peak(&peaks, threshold);
        let chosen_peak = match chosen_peak {
            Some(peak) => {
                 Some(correct_peak(peak, autocorr, PeakCorrection::None))
            },
            None => {
                None
            }
        };

        let pitch = match chosen_peak {
            Some(peak) => {
                let frequency = T::from_usize(sample_rate).unwrap() / peak.0;
                let clarity = peak.1 / autocorr[0];
                Some(Pitch { frequency, clarity })
            },
            None => {
                None
            }
        };
        pitch
    }
}

fn autocorrelation<T>(signal: &mut [Complex<T>], scratch: &mut [Complex<T>], result: &mut [T])
    where T : FloatCore
{
    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(signal.len());
    fft.process(signal, scratch);
    for i in 0..scratch.len() {
        scratch[i].re = scratch[i].re * scratch[i].re + scratch[i].im * scratch[i].im;
        scratch[i].im = T::zero();
    }
    let mut planner = FFTplanner::new(true);
    let inv_fft = planner.plan_fft(signal.len());
    inv_fft.process(scratch, signal);
    copy_complex_to_real(signal, result, ComplexComponent::Re);
}

fn detect_crossings<T: FloatCore>(arr: &[T]) -> Vec<(usize, usize)> {
    let mut crossings = Vec::new();
    let mut positive_zero_cross: Option<usize> = None;
    for i in 1..arr.len() {
        let val = arr[i];
        let prev_val = arr[i - 1];
        match positive_zero_cross {
            Some(idx) => {
                if val < T::zero() && prev_val > T::zero() {
                    crossings.push((idx, i));
                    positive_zero_cross = None;
                }
            },
            None => {
                if val > T::zero() && prev_val < T::zero() {
                    positive_zero_cross = Some(i);
                }
            }
        }
    }
    crossings
}

fn detect_peaks<T: FloatCore>(arr: &[T]) -> Vec<(usize, T)> {
    let crossings = detect_crossings(arr);
    let mut peaks = Vec::new();

    for crossing in crossings {
        let (start, stop) = crossing;
        let mut peak_idx = 0;
        let mut peak_val = - T::infinity();
        for i in start..stop {
            if arr[i] > peak_val {
                peak_val = arr[i];
                peak_idx = i;
            }
        }
        peaks.push((peak_idx, peak_val));
    }

    peaks
}

fn choose_peak<T: FloatCore>(peaks: &[(usize, T)], threshold: T) -> Option<(usize, T)> {
    let mut chosen: Option<(usize, T)> = None;
    for &peak in peaks {
        if peak.1 > threshold {
            chosen = Some(peak);
            break;
        }
    }
    chosen
}

fn correct_peak<T: FloatCore>(peak: (usize, T), data: &[T], correction: PeakCorrection) -> (T, T) {
    match correction {
        PeakCorrection::Quadratic => {
            let idx = peak.0;
            let point = quadratic_interpolation(
                Point{x: T::from_usize(idx - 1).unwrap(), y: data[idx - 1]},
                Point{x: T::from_usize(idx).unwrap(), y: data[idx]},
                Point{x: T::from_usize(idx + 1).unwrap(), y: data[idx + 1]},
            );
            return (point.x, point.y);
        },
        PeakCorrection::None => {
            return (T::from_usize(peak.0).unwrap(), peak.1);
        }
    }
}

fn quadratic_interpolation<T:FloatCore>(left: Point<T>, center: Point<T>, right: Point<T>) -> Point<T> {
    let shift = T::from_f64(0.5).unwrap() * (right.y - left.y) / (T::from_f64(2.0).unwrap() * center.y - left.y - right.y);
    let x = center.x + shift;
    let y = center.y + T::from_f64(0.25).unwrap() * (right.y - left.y) * shift;
    Point { x, y }
}

fn new_real_buffer<T: FloatCore>(size: usize) -> Vec<T> {
    vec![T::zero(); size]
}

fn new_complex_buffer<T: FloatCore>(size: usize) -> Vec<Complex<T>> {
    vec![Complex::zero(); size]
}

fn copy_real_to_complex<T: FloatCore>(input: &[T], output: &mut [Complex<T>], component: ComplexComponent) {
    assert!(input.len() <= output.len());
    match component {
        ComplexComponent::Re => {
            for i in 0..input.len() {
                output[i].re = input[i];
                output[i].im = T::zero();
            }
        },
        ComplexComponent::Im => {
            for i in 0..input.len() {
                output[i].im = input[i];
                output[i].re = T::zero();
            }
        }
    }

    for i in input.len()..output.len() {
        output[i] = Complex::zero();
    }
}

fn copy_complex_to_real<T: FloatCore>(input: &[Complex<T>], output: &mut [T], component: ComplexComponent) {
    assert!(input.len() <= output.len());
    match component {
        ComplexComponent::Re => {
            for i in 0..input.len() {
                output[i] = input[i].re
            }
        },
        ComplexComponent::Im => {
            for i in 0..input.len() {
                output[i] = input[i].im
            }
        }
    }

    for i in input.len()..output.len() {
        output[i] = T::zero();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_chunk<T: FloatCore>(signal: &[T], start: usize, window: usize, output: &mut [T]) {
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

    fn sin_signal<T: FloatCore>(freq: f64, size: usize, sample_rate: usize) -> Vec<T> {
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

    #[test]
    fn peak_correction() {
        let point = quadratic_interpolation(
            Point{x: -1.5, y: - (1.5 * 1.5) + 4.0},
            Point{x: -0.5, y: - (0.5 * 0.5) + 4.0},
            Point{x: 0.5, y: - (0.5 * 0.5) + 4.0}
        );
        assert_eq!(point.x, 0.0);
        assert_eq!(point.y, 4.0);
    }

    #[test]
    fn pure_frequency() {
        const SAMPLE_RATE : usize = 48000;
        const FREQUENCY : f64 = 440.0;
        const DURATION : f64 = 4.0;
        const SAMPLE_SIZE : usize = (SAMPLE_RATE as f64 * DURATION) as usize;
        const WINDOW : usize = 1024;
        const PADDING : usize = WINDOW / 2;
        const DELTA_T : usize = WINDOW / 4;
        const N_WINDOWS : usize = (SAMPLE_SIZE - WINDOW) / DELTA_T;
        const THRESHOLD : f64 = 0.5;

        let signal = sin_signal::<f64>(FREQUENCY, SAMPLE_SIZE, SAMPLE_RATE);

        let mut chunk = new_real_buffer(WINDOW);

        let mut detector = AutocorrelationDetector::<f64>::new(WINDOW, PADDING);

        for i in 0..N_WINDOWS {
            let t : usize = i * DELTA_T;
            get_chunk(&signal, t, WINDOW, &mut chunk);

            let pitch = detector.get_pitch(&chunk, SAMPLE_RATE, THRESHOLD);

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
}
