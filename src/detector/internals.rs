use rustfft::FFTplanner;
use num_complex::Complex;

use crate::float::Float;
use crate::utils::buffer::ComplexComponent;
use crate::utils::buffer::copy_real_to_complex;
use crate::utils::buffer::copy_complex_to_real;
use crate::utils::buffer::new_real_buffer;
use crate::utils::buffer::new_complex_buffer;
use crate::utils::peak::PeakCorrection;
use crate::utils::peak::detect_peaks;
use crate::utils::peak::choose_peak;
use crate::utils::peak::correct_peak;

pub struct Pitch<T>
    where T: Float
{
    pub frequency: T,
    pub clarity: T
}

pub struct DetectorInternals<T>
    where T: Float
{
    pub size: usize,
    pub padding: usize,
    pub real_buffers: Vec<Vec<T>>,
    pub complex_buffers: Vec<Vec<Complex<T>>>
}

impl<T> DetectorInternals<T>
    where T : Float
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

pub fn autocorrelation<T>(signal: &[T], signal_complex: &mut [Complex<T>], scratch: &mut [Complex<T>], result: &mut [T])
    where T : Float
{
    copy_real_to_complex(signal, signal_complex, ComplexComponent::Re);
    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(signal_complex.len());
    fft.process(signal_complex, scratch);
    for i in 0..scratch.len() {
        scratch[i].re = scratch[i].re * scratch[i].re + scratch[i].im * scratch[i].im;
        scratch[i].im = T::zero();
    }
    let mut planner = FFTplanner::new(true);
    let inv_fft = planner.plan_fft(signal_complex.len());
    inv_fft.process(scratch, signal_complex);
    copy_complex_to_real(signal_complex, result, ComplexComponent::Re);
}

pub fn pitch_from_peaks<T>(input: &[T], sample_rate: usize, clarity_threshold: T, correction: PeakCorrection) -> Option<Pitch<T>>
    where T: Float
{
    let peaks = detect_peaks(input);
    let chosen_peak = choose_peak(&peaks, clarity_threshold);
    let chosen_peak = match chosen_peak {
        Some(peak) => {
                Some(correct_peak(peak, input, correction))
        },
        None => {
            None
        }
    };

    let pitch = match chosen_peak {
        Some(peak) => {
            let frequency = T::from_usize(sample_rate).unwrap() / peak.0;
            let clarity = peak.1 / input[0];
            Some(Pitch { frequency, clarity })
        },
        None => {
            None
        }
    };
    pitch
}

pub fn get_power_level<T>(signal: &[T]) -> T
    where T : Float
{
    let mut power = T::zero();
    for i in 0..signal.len() {
        power = power + signal[i] * signal[i];
    }
    power
}

fn m_of_tau<T>(signal: &[T], signal_square_sum: Option<T>, result: &mut [T])
    where T : Float
{
    assert!(result.len() >= signal.len());

    let signal_square_sum = match signal_square_sum {
        Some(val) => val,
        None => {
            let mut val = T::zero();
            for i in 0..signal.len() {
                val = val + signal[i] * signal[i];
            }
            val
        }
    };

    result[0] = T::from_usize(2).unwrap() * signal_square_sum;
    for i in 1..signal.len() {
        result[i] = result[i - 1] - signal[i - 1] * signal[i - 1];
    }

    // Signal has no padding, but result does
    for i in signal.len()..result.len() {
        result[i] = result[i - 1];
    }
}

pub fn normalized_square_difference<T>(signal: &[T], scratch0: &mut [Complex<T>], scratch1: &mut [Complex<T>], scratch2: &mut [T], result: &mut [T])
    where T : Float
{
    autocorrelation(signal, scratch0, scratch1, result);
    m_of_tau(signal, Some(result[0]), scratch2);
    for i in 0..result.len() {
        result[i] = T::from_usize(2).unwrap() * result[i] / scratch2[i];
    }
}
