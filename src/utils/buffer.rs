use num_complex::Complex;
use num_traits::Zero;

use crate::float::Float;

pub enum ComplexComponent {
    Re,
    Im,
}

pub fn new_real_buffer<T: Float>(size: usize) -> Vec<T> {
    vec![T::zero(); size]
}

pub fn new_complex_buffer<T: Float>(size: usize) -> Vec<Complex<T>> {
    vec![Complex::zero(); size]
}

pub fn copy_real_to_complex<T: Float>(
    input: &[T],
    output: &mut [Complex<T>],
    component: ComplexComponent,
) {
    assert!(input.len() <= output.len());
    match component {
        ComplexComponent::Re => input.iter().zip(output.iter_mut()).for_each(|(i, o)| {
            o.re = *i;
            o.im = T::zero();
        }),
        ComplexComponent::Im => input.iter().zip(output.iter_mut()).for_each(|(i, o)| {
            o.im = *i;
            o.re = T::zero();
        }),
    }
    output[input.len()..]
        .iter_mut()
        .for_each(|o| *o = Complex::zero())
}

pub fn copy_complex_to_real<T: Float>(
    input: &[Complex<T>],
    output: &mut [T],
    component: ComplexComponent,
) {
    assert!(input.len() <= output.len());
    match component {
        ComplexComponent::Re => input
            .iter()
            .map(|c| c.re)
            .zip(output.iter_mut())
            .for_each(|(i, o)| *o = i),
        ComplexComponent::Im => input
            .iter()
            .map(|c| c.im)
            .zip(output.iter_mut())
            .for_each(|(i, o)| *o = i),
    }

    output[input.len()..]
        .iter_mut()
        .for_each(|o| *o = T::zero());
}
