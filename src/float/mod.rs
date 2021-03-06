use num_traits::float::FloatCore as NumFloatCore;
use rustfft::FftNum;
use std::fmt::{Debug, Display};

pub trait Float: Display + Debug + NumFloatCore + FftNum {}

impl Float for f64 {}
impl Float for f32 {}
