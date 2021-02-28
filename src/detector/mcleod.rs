use crate::detector::internals::get_power_level;
use crate::detector::internals::normalized_square_difference;
use crate::detector::internals::pitch_from_peaks;
use crate::detector::internals::DetectorInternals;
use crate::detector::internals::Pitch;
use crate::detector::PitchDetector;
use crate::float::Float;
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
        let internals = DetectorInternals::new(2, 2, size, padding);
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
        // We need at least two real and two complex buffers for scratch.
        let real_buffers = &mut self.internals.real_buffers;
        let complex_buffers = &mut self.internals.complex_buffers;
        assert!(real_buffers.len() >= 2);
        assert!(complex_buffers.len() >= 2);

        if get_power_level(signal) < power_threshold {
            return None;
        }

        let (signal_complex, scratch0) = split_first_two_mut(complex_buffers);
        let (scratch1, peaks) = split_first_two_mut(real_buffers);

        normalized_square_difference(signal, signal_complex, scratch0, scratch1, peaks);

        pitch_from_peaks(
            peaks,
            sample_rate,
            clarity_threshold,
            PeakCorrection::Quadratic,
        )
    }
}

/// Split the first two elements from `array` off as mutable elements in a tuple.
fn split_first_two_mut<T>(array: &mut Vec<T>) -> (&mut T, &mut T) {
    let mut iter = array.iter_mut();
    (iter.next().unwrap(), iter.next().unwrap())
}
