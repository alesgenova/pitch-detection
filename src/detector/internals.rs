use rustfft::FftPlanner;

use crate::utils::buffer::ComplexComponent;
use crate::utils::buffer::{copy_complex_to_real, square_sum};
use crate::utils::buffer::{copy_real_to_complex, BufferPool};
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
/// free buffers repeatedly, so we use a `BufferPool` object to manage the buffers.
pub struct DetectorInternals<T>
where
    T: Float,
{
    pub size: usize,
    pub padding: usize,
    pub buffers: BufferPool<T>,
}

impl<T> DetectorInternals<T>
where
    T: Float,
{
    pub fn new(size: usize, padding: usize) -> Self {
        let buffers = BufferPool::new(size + padding);

        DetectorInternals {
            size,
            padding,
            buffers,
        }
    }
}

/// Compute the autocorrelation of `signal` to `result`. All buffers but `signal`
/// may be used as scratch.
pub fn autocorrelation<T>(signal: &[T], buffers: &mut BufferPool<T>, result: &mut [T])
where
    T: Float,
{
    let (ref1, ref2) = (buffers.get_complex_buffer(), buffers.get_complex_buffer());
    let signal_complex = &mut ref1.borrow_mut()[..];
    let scratch = &mut ref2.borrow_mut()[..];

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

pub fn normalized_square_difference<T>(signal: &[T], buffers: &mut BufferPool<T>, result: &mut [T])
where
    T: Float + std::iter::Sum,
{
    let two = T::from_usize(2).unwrap();

    let scratch_ref = buffers.get_real_buffer();
    let scratch = &mut scratch_ref.borrow_mut()[..];

    autocorrelation(signal, buffers, result);
    m_of_tau(signal, Some(result[0]), scratch);
    result
        .iter_mut()
        .zip(scratch)
        .for_each(|(r, s)| *r = two * *r / *s)
}

/// Compute the windowed autocorrelation of `signal` and put the result in `result`.
/// For a signal _x=(x_0,x_1,...)_, the windowed autocorrelation with window size _w_ is
/// the function
///
/// > r(t) = sum_{i=0}^{w-1} x_i*x_{i+t}
///
/// This function assumes `window_size` is at most half of the length of `signal`.
pub fn windowed_autocorrelation<T>(
    signal: &[T],
    window_size: usize,
    buffers: &mut BufferPool<T>,
    result: &mut [T],
) where
    T: Float + std::iter::Sum,
{
    assert!(
        buffers.buffer_size >= signal.len(),
        "Buffers must have a length at least equal to `signal`."
    );

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(signal.len());
    let inv_fft = planner.plan_fft_inverse(signal.len());

    let (scratch_ref1, scratch_ref2, scratch_ref3) = (
        buffers.get_complex_buffer(),
        buffers.get_complex_buffer(),
        buffers.get_complex_buffer(),
    );

    let signal_complex = &mut scratch_ref1.borrow_mut()[..signal.len()];
    let truncated_signal_complex = &mut scratch_ref2.borrow_mut()[..signal.len()];
    let scratch = &mut scratch_ref3.borrow_mut()[..signal.len()];

    // To achieve the windowed autocorrelation, we compute the cross correlation between
    // the original signal and the signal truncated to lie in `0..window_size`
    copy_real_to_complex(signal, signal_complex, ComplexComponent::Re);
    copy_real_to_complex(
        &signal[..window_size],
        truncated_signal_complex,
        ComplexComponent::Re,
    );
    fft.process_with_scratch(signal_complex, scratch);
    fft.process_with_scratch(truncated_signal_complex, scratch);
    // rustfft doesn't normalize when it computes the fft, so we need to normalize ourselves by
    // dividing by `sqrt(signal.len())` each time we take an fft or inverse fft.
    // Since the fft is linear and we are doing fft -> inverse fft, we can just divide by
    // `signal.len()` once.
    let normalization_const = T::one() / T::from_usize(signal.len()).unwrap();
    signal_complex
        .iter_mut()
        .zip(truncated_signal_complex.iter())
        .for_each(|(a, b)| {
            *a = *a * normalization_const * b.conj();
        });
    inv_fft.process_with_scratch(signal_complex, scratch);

    // The result is valid only for `0..window_size`
    copy_complex_to_real(&signal_complex[..window_size], result, ComplexComponent::Re);
}

/// Compute the windowed square error, _d(t)_, of `signal`. For a window size of _w_ and a signal
/// _x=(x_0,x_1,...)_, this is defined by
///
///  > d(t) = sum_{i=0}^{w-1} (x_i - x_{i+t})^2
///
/// This function is computed efficiently using an FFT. It is assumed that `window_size` is at most half
/// the length of `signal`.
pub fn windowed_square_error<T>(
    signal: &[T],
    window_size: usize,
    buffers: &mut BufferPool<T>,
    result: &mut [T],
) where
    T: Float + std::iter::Sum,
{
    assert!(
        2 * window_size <= signal.len(),
        "The window size cannot be more than half the signal length"
    );

    let two = T::from_f64(2.).unwrap();

    // The windowed square error function, d(t), can be computed
    // as d(t) = pow_0^w + pow_t^{t+w} - 2*windowed_autocorrelation(t)
    // where pow_a^b is the sum of the square of `signal` on the window `a..b`
    // We proceed accordingly.
    windowed_autocorrelation(signal, window_size, buffers, result);
    let mut windowed_power = square_sum(&signal[..window_size]);
    let power = windowed_power;

    result.iter_mut().enumerate().for_each(|(i, a)| {
        // use the formula pow_0^w + pow_t^{t+w} - 2*windowed_autocorrelation(t)
        *a = power + windowed_power - two * *a;
        // Since we're processing everything in order, we can computed pow_{t+1}^{t+1+w}
        // directly from pow_t^{t+w} by adding and subtracting the boundary terms.
        windowed_power = windowed_power - signal[i] * signal[i]
            + signal[i + window_size] * signal[i + window_size];
    })
}

/// Calculate the "cumulative mean normalized difference function" as
/// specified in the YIN paper. If _d(t)_ is the square error function,
/// compute _d'(0) = 1_ and for _t > 0_
///
///  > d'(t) = d(t) / [ (1/t) * sum_{i=0}^t d(i) ]
pub fn yin_normalize_square_error<T: Float>(square_error: &mut [T]) {
    let mut sum = T::zero();
    square_error[0] = T::one();
    // square_error[0] should always be zero, so we don't need to worry about
    // adding this to our sum.
    square_error
        .iter_mut()
        .enumerate()
        .skip(1)
        .for_each(|(i, a)| {
            sum = sum + *a;
            *a = *a * T::from_usize(i + 1).unwrap() / sum;
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windowed_autocorrelation_test() {
        let signal: Vec<f64> = vec![0., 1., 2., 0., -1., -2.];
        let window_size: usize = 3;

        let buffers = &mut BufferPool::new(signal.len());

        let result: Vec<f64> = (0..window_size)
            .map(|i| {
                signal[..window_size]
                    .iter()
                    .zip(signal[i..(i + window_size)].iter())
                    .map(|(a, b)| *a * *b)
                    .sum()
            })
            .collect();

        let mut computed_result = vec![0.; window_size];
        windowed_autocorrelation(&signal, window_size, buffers, &mut computed_result);
        // Using an FFT loses precision; we don't care that much, so round generously.
        computed_result
            .iter_mut()
            .for_each(|x| *x = (*x * 100.).round() / 100.);

        assert_eq!(result, computed_result);
    }

    #[test]
    fn windowed_square_error_test() {
        let signal: Vec<f64> = vec![0., 1., 2., 0., -1., -2.];
        let window_size: usize = 3;

        let buffers = &mut BufferPool::new(signal.len());

        let result: Vec<f64> = (0..window_size)
            .map(|i| {
                signal[..window_size]
                    .iter()
                    .zip(signal[i..(i + window_size)].iter())
                    .map(|(x_j, x_j_tau)| (*x_j - *x_j_tau) * (*x_j - *x_j_tau))
                    .sum()
            })
            .collect();

        let mut computed_result = vec![0.; window_size];
        windowed_square_error(&signal, window_size, buffers, &mut computed_result);
        // Using an FFT loses precision; we don't care that much, so round generously.
        computed_result
            .iter_mut()
            .for_each(|x| *x = (*x * 100.).round() / 100.);

        assert_eq!(result, computed_result);
    }
    #[test]
    fn yin_normalized_square_error_test() {
        let signal: &mut Vec<f64> = &mut vec![0., 6., 14.];
        let result = vec![1., 2., 3. * 14. / (6. + 14.)];

        yin_normalize_square_error(signal);

        assert_eq!(result, *signal);
    }
}
