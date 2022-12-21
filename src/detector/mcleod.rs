//! The McLeod pitch detection algorithm is based on the algorithm from the paper
//! *[A Smarter Way To Find Pitch](https://www.researchgate.net/publication/230554927_A_smarter_way_to_find_pitch)*.
//! It is efficient and offers an improvement over basic autocorrelation.
//!
//! The algorithm is based on finding peaks of the *normalized square difference* function. Let $S=(s_0,s_1,\ldots,s_N)$
//! be a discrete signal. The *square difference function* at time $t$ is defined by
//! $$ d\'(t) = \sum_{i=0}^{N-t} (s_i-s_{i+t})^2. $$
//! This function is close to zero when the signal "lines up" with itself. However, *close* is a relative term,
//! and the value of $d\'(t)$ depends on volume, which should not affect the pitch of the signal. For this
//! reason, the *normalized square difference function*, $n\'(t)$, is computed.
//! $$ n\'(t) = \frac{d\'(t)}{\sum_{i=0}^{N-t} (x_i^2+x_{i+t}^2) } $$
//! The algorithm then searches for the first local minimum of $n\'(t)$ below a given threshold, called the
//! *clarity threshold*.
//!
//! ## Implementation
//! As outlined in *A Smarter Way To Find Pitch*,
//! an [FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform) is used to greatly speed up the computation of
//! the normalized square difference function. Further, the algorithm applies some algebraic tricks and actually
//! searches for the *peaks* of $1-n\'(t)$, rather than minimums of $n\'(t)$.
//!
//! After a peak is found, quadratic interpolation is applied to further refine the estimate.
use crate::detector::internals::normalized_square_difference;
use crate::detector::internals::pitch_from_peaks;
use crate::detector::internals::DetectorInternals;
use crate::detector::internals::Pitch;
use crate::detector::PitchDetector;
use crate::float::Float;
use crate::utils::buffer::square_sum;
use crate::utils::peak::PeakCorrection;

pub struct McLeodDetector<T>
where
    T: Float + std::iter::Sum,
{
    internals: DetectorInternals<T>,
}

impl<T> McLeodDetector<T>
where
    T: Float + std::iter::Sum,
{
    pub fn new(size: usize, padding: usize) -> Self {
        let internals = DetectorInternals::new(size, padding);
        McLeodDetector { internals }
    }
}

impl<T> PitchDetector<T> for McLeodDetector<T>
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
        let result = &mut result_ref.write().unwrap()[..];

        normalized_square_difference(signal, &mut self.internals.buffers, result);
        pitch_from_peaks(
            result,
            sample_rate,
            clarity_threshold,
            PeakCorrection::Quadratic,
        )
    }
}
