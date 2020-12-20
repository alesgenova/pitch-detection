use num_complex::Complex;
use num_traits::Zero;
use rustfft::FFTplanner;

use crate::detector::internals::Pitch;
use crate::detector::PitchDetector;
use crate::detector::PolyphonicDetector;
use crate::float::Float;
use crate::utils::buffer::copy_complex_to_real;
use crate::utils::buffer::copy_real_to_complex;
use crate::utils::buffer::new_complex_buffer;
use crate::utils::buffer::new_real_buffer;
use crate::utils::buffer::ComplexComponent;
use crate::utils::filters::comb_filter;

pub struct CombDetector<T>
where
    T: Float,
{
    detector: Box<dyn PitchDetector<T>>,
    limit: usize,
}

impl<T> CombDetector<T>
where
    T: Float,
{
    pub fn new(detector: Box<dyn PitchDetector<T>>, limit: usize) -> Self {
        CombDetector { detector, limit }
    }
}

impl<T> PolyphonicDetector<T> for CombDetector<T>
where
    T: Float,
{
    fn get_pitch(
        &mut self,
        signal: &[T],
        sample_rate: usize,
        power_threshold: T,
        clarity_threshold: T,
    ) -> Vec<Option<Pitch<T>>> {
        let mut pitches = vec![None; self.limit];
        let size = signal.len();
        let size_t = T::from_usize(size).unwrap();

        let mut signal_complex: Vec<Complex<T>> = new_complex_buffer(size, Complex::zero());
        let mut scratch: Vec<Complex<T>> = new_complex_buffer(size, Complex::zero());
        let mut comb_g: Vec<T> = new_real_buffer(signal.len(), T::one());
        let mut remaining_signal: Vec<T> = signal.to_vec();

        let beta = T::from_f64(50.).unwrap(); // TODO: expose this as a paremeter

        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(size);
        let mut planner = FFTplanner::new(true);
        let inv_fft = planner.plan_fft(size);

        for iter in 0..self.limit {
            let pitch = self.detector.get_pitch(
                &remaining_signal,
                sample_rate,
                power_threshold,
                clarity_threshold,
            );

            let stop = match pitch {
                Some(p) => {
                    pitches[iter] = Some(p.clone());

                    copy_real_to_complex(
                        &remaining_signal,
                        &mut signal_complex,
                        ComplexComponent::Re,
                    );
                    fft.process(&mut signal_complex, &mut scratch);
                    comb_filter(p.frequency, beta, size, sample_rate, &mut comb_g);

                    scratch
                        .iter_mut()
                        .zip(comb_g.iter())
                        .for_each(|(s_value, c_value)| {
                            s_value.re = s_value.re * (*c_value) / size_t;
                            s_value.im = s_value.im * (*c_value) / size_t;
                        });

                    inv_fft.process(&mut scratch, &mut signal_complex);
                    copy_complex_to_real(
                        &signal_complex,
                        &mut remaining_signal,
                        ComplexComponent::Re,
                    );

                    false
                }
                None => true,
            };

            if stop {
                return pitches;
            }
        }

        pitches
    }
}
