//! Autocorrelation is one of the most basic forms of pitch detection. Let $S=(s_0,s_1,\ldots,s_N)$
//! be a discrete signal. Then, the autocorrelation function of $S$ at time $t$ is
//! $$ A_t(S) = \sum_{i=0}^{N-t} s_i s_{i+t}. $$
//! The autocorrelation function is largest when $t=0$. Subsequent peaks indicate when the signal
//! is particularly well aligned with itself. Thus, peaks of $A_t(S)$ when $t>0$ are good candidates
//! for the fundamental frequency of $S$.
//!
//! Unfortunately, autocorrelation-based pitch detection is prone to octave errors, since a signal
//! may "line up" with itself better when shifted by amounts larger than by the fundamental frequency.
//! Further, autocorrelation is a bad choice for situations where the fundamental frequency may not
//! be the loudest frequency (which is common in telephone speech and for certain types of instruments).
//!
//! ## Implementation
//! Rather than compute the autocorrelation function directly, an [FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform)
//! is used, providing a dramatic speed increase for large buffers.

use crate::detector::internals::pitch_from_peaks;
use crate::detector::internals::DetectorInternals;
use crate::detector::internals::Pitch;
use crate::detector::PitchDetector;
use crate::float::Float;
use crate::utils::peak::PeakCorrection;
use crate::{detector::internals::autocorrelation, utils::buffer::square_sum};

pub struct AutocorrelationDetector<T>
where
    T: Float,
{
    internals: DetectorInternals<T>,
}

impl<T> AutocorrelationDetector<T>
where
    T: Float,
{
    pub fn new(size: usize, padding: usize) -> Self {
        let internals = DetectorInternals::new(size, padding);
        AutocorrelationDetector { internals }
    }
}

impl<T> PitchDetector<T> for AutocorrelationDetector<T>
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
        assert_eq!(signal.len(), self.internals.size);

        if square_sum(signal) < power_threshold {
            return None;
        }

        let result_ref = self.internals.buffers.get_real_buffer();
        let result = &mut result_ref.borrow_mut()[..];

        autocorrelation(signal, &mut self.internals.buffers, result);
        let clarity_threshold = clarity_threshold * result[0];

        pitch_from_peaks(result, sample_rate, clarity_threshold, PeakCorrection::None)
    }
}
