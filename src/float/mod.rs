use num_traits::float::FloatCore as NumFloatCore;
use rustfft::FFTnum;
use std::fmt::Display;

pub trait Float: Display + NumFloatCore + FFTnum {}

impl Float for f64 {}
impl Float for f32 {}
