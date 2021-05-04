use object_pool::{Pool, Reusable};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

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

/// Computes |x|^2 for each complex value x in `arr`. This function
/// modifies `arr` in place and leaves the complex component zero.
pub fn modulus_squared<'a, T: Float>(arr: &'a mut [Complex<T>]) {
    for mut s in arr {
        s.re = s.re * s.re + s.im * s.im;
        s.im = T::zero();
    }
}

/// Compute the sum of the square of each element of `arr`.
pub fn square_sum<T>(arr: &[T]) -> T
where
    T: Float + std::iter::Sum,
{
    arr.iter().map(|&s| s * s).sum::<T>()
}

/// A pool of real/complex buffer objects. Buffers are dynamically created as needed
/// and reused if previously `Drop`ed. Buffers are never freed. Instead buffers are kept
/// in reserve and reused when a new buffer is requested.
///
/// ```rust
/// use pitch_detection::utils::buffer::BufferPool;
///
/// let buffers = BufferPool::new(3);
/// let mut buf1 = buffers.get_real_buffer();
/// {
///     // This buffer won't be dropped until the end of the function
///     buf1[0] = 5.5;
/// }
/// {
///     // This buffer will be dropped when the scope ends
///     let mut buf2 = buffers.get_real_buffer();
///     buf2[1] = 6.6;
/// }
/// {
///     // This buffer will be dropped when the scope ends
///     // It is the same buffer that was just used (i.e., it's a reused buffer)
///     let mut buf3 = buffers.get_real_buffer();
///     buf3[2] = 7.7;
/// }
/// drop(buf1);
///
/// let buf1 = &buffers.get_real_buffer();
/// let buf2 = &buffers.get_real_buffer();
/// // Buffers are distributed in LIFO order, so compare them "backwards".
/// assert_eq!(&buf2[..], &[0.0, 6.6, 7.7]);
/// assert_eq!(&buf1[..], &[5.5, 0., 0.]);
/// ```
pub struct BufferPool<T> {
    real_buffers: Pool<Vec<T>>,
    complex_buffers: Pool<Vec<Complex<T>>>,
    pub buffer_size: usize,
}

impl<T: Float> BufferPool<T> {
    pub fn new(buffer_size: usize) -> Self {
        BufferPool {
            real_buffers: Pool::new(0, || new_real_buffer(buffer_size)),
            complex_buffers: Pool::new(0, || new_complex_buffer(buffer_size)),
            buffer_size,
        }
    }
    /// Get a reference to a buffer that can be used until it is `Drop`ed.
    pub fn get_real_buffer(&self) -> Reusable<Vec<T>> {
        self.real_buffers.pull(|| new_real_buffer(self.buffer_size))
    }
    /// Get a reference to a buffer that can be used until it is `Drop`ed.
    pub fn get_complex_buffer(&self) -> Reusable<Vec<Complex<T>>> {
        self.complex_buffers
            .pull(|| new_complex_buffer(self.buffer_size))
    }
}

#[test]
fn test_buffers() {
    let buffers = BufferPool::new(3);
    let mut buf1 = buffers.get_real_buffer();
    {
        // This buffer won't be dropped until the end of the function
        // or a manual call to `drop`.
        buf1[0] = 5.5;
    }
    {
        // This buffer will be dropped when the scope ends
        let mut buf2 = buffers.get_real_buffer();
        buf2[1] = 6.6;
    }
    {
        // This buffer will be dropped when the scope ends
        // It is the same buffer that was just used (i.e., it's a reused buffer)
        let mut buf3 = buffers.get_real_buffer();
        buf3[2] = 7.7;
    }
    drop(buf1);

    let buf1 = &buffers.get_real_buffer();
    let buf2 = &buffers.get_real_buffer();
    // Buffers are distributed in LIFO order, so compare them "backwards".
    assert_eq!(&buf2[..], &[0.0, 6.6, 7.7]);
    assert_eq!(&buf1[..], &[5.5, 0., 0.]);
}
