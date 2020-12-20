use num_traits::float::FloatCore as NumFloatCore;
use rustfft::FFTnum;

pub trait Number {
    fn exp(&self) -> Self;
}

impl Number for f64 {
    fn exp(&self) -> Self {
        f64::exp(*self)
    }
}

impl Number for f32 {
    fn exp(&self) -> Self {
        f32::exp(*self)
    }
}

pub trait Float: NumFloatCore + FFTnum + Number {}

impl Float for f64 {}
impl Float for f32 {}
