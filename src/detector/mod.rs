use crate::float::Float;
use crate::detector::internals::Pitch;

pub mod internals;
pub mod autocorrelation;
pub mod mcleod;

pub trait PitchDetector<T>
    where T : Float
{
    fn get_pitch(&mut self, signal: &[T], sample_rate: usize, power_threshold: T, clarity_threshold: T) -> Option<Pitch<T>>;
}
