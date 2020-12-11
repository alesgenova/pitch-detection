use crate::detector::internals::autocorrelation;
use crate::detector::internals::get_power_level;
use crate::detector::internals::pitch_from_peaks;
use crate::detector::internals::DetectorInternals;
use crate::detector::internals::Pitch;
use crate::detector::PitchDetector;
use crate::float::Float;
use crate::utils::peak::PeakCorrection;

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
        let internals = DetectorInternals::new(1, 2, size, padding);
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

        if get_power_level(signal) < power_threshold {
            return None;
        }

        let (signal_complex, rest) = self.internals.complex_buffers.split_first_mut().unwrap();
        let (scratch, _) = rest.split_first_mut().unwrap();
        let (autocorr, _) = self.internals.real_buffers.split_first_mut().unwrap();

        autocorrelation(signal, signal_complex, scratch, autocorr);
        let clarity_threshold = clarity_threshold * autocorr[0];

        pitch_from_peaks(
            autocorr,
            sample_rate,
            clarity_threshold,
            PeakCorrection::None,
        )
    }
}
