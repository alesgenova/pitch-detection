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

        if get_power_level(signal) < power_threshold {
            return None;
        }

        let (signal_complex, rest) = self.internals.complex_buffers.split_first_mut().unwrap();
        let (scratch0, _) = rest.split_first_mut().unwrap();

        let (scratch1, rest) = self.internals.real_buffers.split_first_mut().unwrap();
        let (nsdf, _) = rest.split_first_mut().unwrap();

        normalized_square_difference(signal, signal_complex, scratch0, scratch1, nsdf);

        pitch_from_peaks(
            nsdf,
            sample_rate,
            clarity_threshold,
            PeakCorrection::Quadratic,
        )
    }
}
