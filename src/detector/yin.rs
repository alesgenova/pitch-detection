//! The YIN pitch detection algorithm is based on the algorithm from the paper
//! *[YIN, a fundamental frequency estimator for speech and music](http://recherche.ircam.fr/equipes/pcm/cheveign/ps/2002_JASA_YIN_proof.pdf)*.
//! It is efficient and offers an improvement over basic autocorrelation.
//!
//! The YIN pitch detection algorithm is similar to the [McLeod][crate::detector::mcleod], but it is based on
//! a different normalization of the *mean square difference function*.
//!
//! Let $S=(s_0,s_1,\ldots,s_N)$ be a discrete signal. The *mean square difference function* at time $t$
//! is defined by
//! $$ d(t) = \sum_{i=0}^{N-t} (s_i-s_{i+t})^2. $$
//! This function is close to zero when the signal "lines up" with itself. However, *close* is a relative term,
//! and the value of $d\'(t)$ depends on volume, which should not affect the pitch of the signal. For this
//! reason, the signal is normalized. The YIN algorithm computes the *cumulative mean normalized difference function*,
//! $$ d\'(t) = \begin{cases}1&\text{if }t=0\\\\ d(t) / \left[ \tfrac{1}{t}\sum_{i=0}^t d(i) \right] & \text{otherwise}\end{cases}. $$
//! Then, it searches for the first local minimum of $d\'(t)$ below a given threshold.
//!
//! ## Implementation
//! Rather than compute the cumulative mean normalized difference function directly,
//! an [FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform) is used, providing a dramatic speed increase for large buffers.
//!
//! After a candidate frequency is found, quadratic interpolation is applied to further refine the estimate.
//!
//! The current implementation does not perform *Step 6* of the algorithm specified in the YIN paper.

use crate::detector::internals::pitch_from_peaks;
use crate::detector::internals::Pitch;
use crate::detector::PitchDetector;
use crate::float::Float;
use crate::utils::buffer::square_sum;
use crate::utils::peak::PeakCorrection;

use super::internals::{windowed_square_error, yin_normalize_square_error, DetectorInternals};

pub struct YINDetector<T>
where
    T: Float + std::iter::Sum,
{
    internals: DetectorInternals<T>,
}

impl<T> YINDetector<T>
where
    T: Float + std::iter::Sum,
{
    pub fn new(size: usize, padding: usize) -> Self {
        let internals = DetectorInternals::<T>::new(size, padding);
        YINDetector { internals }
    }
}

/// Pitch detection based on the YIN algorithm. See <http://recherche.ircam.fr/equipes/pcm/cheveign/ps/2002_JASA_YIN_proof.pdf>
impl<T> PitchDetector<T> for YINDetector<T>
where
    T: Float + std::iter::Sum,
{
    fn get_pitch(
        &mut self,
        signal: &[T],
        sample_rate: usize,
        power_threshold: T,
        clarity_threshold: T,
    ) -> Option<Pitch<T>> {
        // The YIN paper uses 0.1 as a threshold; TarsosDSP uses 0.2. `threshold` is not quite
        // the same thing as 1 - clarity, but it should be close enough.
        let threshold = T::one() - clarity_threshold;
        let window_size = signal.len() / 2;

        assert_eq!(signal.len(), self.internals.size);

        if square_sum(signal) < power_threshold {
            return None;
        }

        let result_ref = self.internals.buffers.get_real_buffer();
        let result = &mut result_ref.borrow_mut()[..window_size];

        // STEP 2: Calculate the difference function, d_t.
        windowed_square_error(signal, window_size, &mut self.internals.buffers, result);

        // STEP 3: Calculate the cumulative mean normalized difference function, d_t'.
        yin_normalize_square_error(result);

        // STEP 4: The absolute threshold. We want the first dip below `threshold`.
        // The YIN paper looks for minimum peaks. Since `pitch_from_peaks` looks
        // for maximums, we take this opportunity to invert the signal.
        result.iter_mut().for_each(|val| *val = threshold - *val);

        // STEP 5: Find the peak and use quadratic interpolation to fine-tune the result
        pitch_from_peaks(result, sample_rate, T::zero(), PeakCorrection::Quadratic).map(|pitch| {
            Pitch {
                frequency: pitch.frequency,
                // A `clarity` is not given by the YIN algorithm. However, we can
                // say a pitch has higher clarity if it's YIN normalized square error is closer to zero.
                // We can then take 1 - YIN error and report that as `clarity`.
                clarity: T::one() - threshold + pitch.clarity / threshold,
            }
        })

        // STEP 6: TODO. Step 6 of the YIN paper can eek out a little more accuracy/consistency, but
        // it also involves computing over a much larger window.
    }
}
