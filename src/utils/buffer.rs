use num_traits::Zero;
use num_complex::Complex;

use crate::float::Float;

pub enum ComplexComponent {
  Re,
  Im
}

pub fn new_real_buffer<T: Float>(size: usize) -> Vec<T> {
  vec![T::zero(); size]
}

pub fn new_complex_buffer<T: Float>(size: usize) -> Vec<Complex<T>> {
  vec![Complex::zero(); size]
}

pub fn copy_real_to_complex<T: Float>(input: &[T], output: &mut [Complex<T>], component: ComplexComponent) {
  assert!(input.len() <= output.len());
  match component {
      ComplexComponent::Re => {
          for i in 0..input.len() {
              output[i].re = input[i];
              output[i].im = T::zero();
          }
      },
      ComplexComponent::Im => {
          for i in 0..input.len() {
              output[i].im = input[i];
              output[i].re = T::zero();
          }
      }
  }

  for i in input.len()..output.len() {
      output[i] = Complex::zero();
  }
}

pub fn copy_complex_to_real<T: Float>(input: &[Complex<T>], output: &mut [T], component: ComplexComponent) {
  assert!(input.len() <= output.len());
  match component {
      ComplexComponent::Re => {
          for i in 0..input.len() {
              output[i] = input[i].re
          }
      },
      ComplexComponent::Im => {
          for i in 0..input.len() {
              output[i] = input[i].im
          }
      }
  }

  for i in input.len()..output.len() {
      output[i] = T::zero();
  }
}
