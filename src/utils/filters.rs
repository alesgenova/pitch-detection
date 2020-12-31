use crate::float::Float;

pub fn comb_filter<T>(
    freq: T,
    beta: T,
    size: usize,
    sample_rate: usize,
    result: &mut [T],
    lower_octaves: bool,
    higher_octaves: bool,
    initialize: bool,
) where
    T: Float,
{
    assert_eq!(size, result.len());

    let one = T::one();
    let two = one + one;

    let dh = T::from_usize(size).unwrap() / T::from_usize(sample_rate).unwrap();

    let mut nodes = vec![];
    nodes.push(freq);

    if lower_octaves {
        let min_freq = one / dh;
        let mut f = freq / two;
        while f > min_freq {
            nodes.push(f);
            f = f / two;
        }
    }

    if higher_octaves {
        let max_freq = T::from_usize(size / 2).unwrap() / dh;
        let mut f = freq * two;
        while f < max_freq {
            nodes.push(f);
            f = f * two;
        }
    }

    if initialize {
        result.iter_mut().for_each(|val| {
            *val = one;
        });
    }

    for idx in 0..(size / 2) {
        let i = idx + 1;
        let f = T::from_usize(i).unwrap() / dh;

        for node in nodes.iter() {
            let mut arg = (f - *node).abs() / beta;
            arg = arg * arg;
            let damp = one - (-arg).exp();
            result[i] = result[i] * damp;
        }
    }

    for idx in 0..(size / 2) {
        let i = idx + 1;
        let j = size - idx - 1;
        result[j] = result[i];
    }
}
