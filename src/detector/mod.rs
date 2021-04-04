use crate::detector::internals::Pitch;
use crate::float::Float;

pub mod autocorrelation;
pub mod internals;
pub mod mcleod;
pub mod yin;

pub trait PitchDetector<T>
where
    T: Float,
{
    fn get_pitch(
        &mut self,
        signal: &[T],
        sample_rate: usize,
        power_threshold: T,
        clarity_threshold: T,
    ) -> Option<Pitch<T>>;
}
