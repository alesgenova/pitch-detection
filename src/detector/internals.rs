use num_complex::Complex;
use rustfft::FftPlanner;

use crate::utils::buffer::copy_real_to_complex;
use crate::utils::buffer::new_complex_buffer;
use crate::utils::buffer::new_real_buffer;
use crate::utils::buffer::ComplexComponent;
use crate::utils::buffer::{copy_complex_to_real, square_sum};
use crate::utils::peak::choose_peak;
use crate::utils::peak::correct_peak;
use crate::utils::peak::detect_peaks;
use crate::utils::peak::PeakCorrection;
use crate::{float::Float, utils::buffer::modulus_squared};

pub struct Pitch<T>
where
    T: Float,
{
    pub frequency: T,
    pub clarity: T,
}

/// Data structure to hold any buffers needed for pitch computation.
/// For WASM it's best to allocate buffers once rather than allocate and
/// free buffers repeatedly.
pub struct DetectorInternals<T>
where
    T: Float,
{
    pub size: usize,
    pub padding: usize,
    pub real_buffers: Vec<Vec<T>>,
    pub complex_buffers: Vec<Vec<Complex<T>>>,
}

impl<T> DetectorInternals<T>
where
    T: Float,
{
    pub fn new(
        n_real_buffers: usize,
        n_complex_buffers: usize,
        size: usize,
        padding: usize,
    ) -> Self {
        let real_buffers: Vec<Vec<T>> = (0..n_real_buffers)
            .map(|_| new_real_buffer(size + padding))
            .collect();

        let complex_buffers: Vec<Vec<Complex<T>>> = (0..n_complex_buffers)
            .map(|_| new_complex_buffer(size + padding))
            .collect();

        DetectorInternals {
            size,
            padding,
            real_buffers,
            complex_buffers,
        }
    }

    // Check whether there are at least the appropriate number of real and complex buffers.
    pub fn has_sufficient_buffers(&self, n_real_buffers: usize, n_complex_buffers: usize) -> bool {
        self.real_buffers.len() >= n_real_buffers && self.complex_buffers.len() >= n_complex_buffers
    }
}

/// Compute the autocorrelation of `signal` to `result`. All buffers but `signal`
/// may be used as scratch.
pub fn autocorrelation<T>(
    signal: &[T],
    signal_complex: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
    result: &mut [T],
) where
    T: Float,
{
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(signal_complex.len());
    let inv_fft = planner.plan_fft_inverse(signal_complex.len());

    // Compute the autocorrelation
    copy_real_to_complex(signal, signal_complex, ComplexComponent::Re);
    fft.process_with_scratch(signal_complex, scratch);
    modulus_squared(signal_complex);
    inv_fft.process_with_scratch(signal_complex, scratch);
    copy_complex_to_real(signal_complex, result, ComplexComponent::Re);
}

pub fn pitch_from_peaks<T>(
    input: &[T],
    sample_rate: usize,
    clarity_threshold: T,
    correction: PeakCorrection,
) -> Option<Pitch<T>>
where
    T: Float,
{
    let sample_rate = T::from_usize(sample_rate).unwrap();
    let peaks = detect_peaks(input);

    choose_peak(peaks, clarity_threshold)
        .map(|peak| correct_peak(peak, input, correction))
        .map(|peak| Pitch {
            frequency: sample_rate / peak.0,
            clarity: peak.1 / input[0],
        })
}

fn m_of_tau<T>(signal: &[T], signal_square_sum: Option<T>, result: &mut [T])
where
    T: Float + std::iter::Sum,
{
    assert!(result.len() >= signal.len());

    let signal_square_sum = signal_square_sum.unwrap_or_else(|| square_sum(signal));

    let start = T::from_usize(2).unwrap() * signal_square_sum;
    result[0] = start;
    let last = result[1..]
        .iter_mut()
        .zip(signal)
        .fold(start, |old, (r, &s)| {
            *r = old - s * s;
            *r
        });
    // Pad the end of `result` with the last value
    result[signal.len()..].iter_mut().for_each(|r| *r = last);
}

pub fn normalized_square_difference<T>(
    signal: &[T],
    scratch0: &mut [Complex<T>],
    scratch1: &mut [Complex<T>],
    scratch2: &mut [T],
    result: &mut [T],
) where
    T: Float + std::iter::Sum,
{
    let two = T::from_usize(2).unwrap();

    autocorrelation(signal, scratch0, scratch1, result);
    m_of_tau(signal, Some(result[0]), scratch2);
    result
        .iter_mut()
        .zip(scratch2)
        .for_each(|(r, s)| *r = two * *r / *s)
}
