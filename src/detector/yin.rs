use crate::detector::internals::pitch_from_peaks;
use crate::detector::internals::DetectorInternals;
use crate::detector::internals::Pitch;
use crate::detector::PitchDetector;
use crate::float::Float;
use crate::utils::buffer::square_sum;
use crate::utils::peak::PeakCorrection;

use super::internals::{windowed_square_error, yin_normalize_square_error};

/// Pitch detection based on the YIN algorithm. See http://recherche.ircam.fr/equipes/pcm/cheveign/ps/2002_JASA_YIN_proof.pdf
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
        let internals = DetectorInternals::new(1, 3, size, padding);
        YINDetector { internals }
    }
}

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
        assert!(
            self.internals.has_sufficient_buffers(1, 3),
            "YINDetector requires at least 1 real and 3 complex buffers"
        );

        if square_sum(signal) < power_threshold {
            return None;
        }

        let mut iter = self.internals.complex_buffers.iter_mut();
        let scratch1 = iter.next().unwrap();
        let scratch2 = iter.next().unwrap();
        let scratch3 = iter.next().unwrap();

        let mut iter = self.internals.real_buffers.iter_mut();
        let result = &mut iter.next().unwrap()[..window_size];

        // STEP 2: Calculate the difference function, d_t.
        windowed_square_error(signal, window_size, (scratch1, scratch2, scratch3), result);

        // STEP 3: Calculate the cumulative mean normalized difference function, d_t'.
        yin_normalize_square_error(result);

        // STEP 4: The absolute threshold. We want the first dip below `threshold`.
        // The YIN paper looks for minimum peaks. Since `pitch_from_peaks` looks
        // for maximums, we take this opportunity to invert the signal.
        result.iter_mut().for_each(|val| *val = threshold - *val);

        // STEP 5: Find the peak and use quadratic interpolation to fine-tune the result
        pitch_from_peaks(result, sample_rate, T::zero(), PeakCorrection::Quadratic)
    }
}
