use num_traits::float::FloatCore as NumFloatCore;
use rustfft::FFTnum;

pub trait Float: NumFloatCore + FFTnum {}

impl Float for f64 {}
impl Float for f32 {}
