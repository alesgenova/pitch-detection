//! # Pitch Detectors
//! Each detector implements a different pitch-detection algorithm.
//! Every detector implements the standard [PitchDetector] trait.

use crate::detector::internals::Pitch;
use crate::float::Float;

pub mod autocorrelation;
#[doc(hidden)]
pub mod internals;
pub mod mcleod;
pub mod yin;

/// A uniform interface to all pitch-detection algorithms.
pub trait PitchDetector<T>
where
    T: Float,
{
    /// Get an estimate of the [Pitch] of the sound sample stored in `signal`.
    ///
    /// Arguments:
    ///
    /// * `signal`: The signal to be analyzed
    /// * `sample_rate`: The number of samples per second contained in the signal.
    /// * `power_threshold`: If the signal has a power below this threshold, no
    ///   attempt is made to find its pitch and `None` is returned.
    /// * `clarity_threshold`: A number between 0 and 1 reflecting the confidence
    ///   the algorithm has in its estimate of the frequency. Higher `clarity_threshold`s
    ///   correspond to higher confidence.
    fn get_pitch(
        &mut self,
        signal: &[T],
        sample_rate: usize,
        power_threshold: T,
        clarity_threshold: T,
    ) -> Option<Pitch<T>>;
}
