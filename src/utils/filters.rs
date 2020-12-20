use crate::float::Float;

pub fn comb_filter<T>(freq: T, beta: T, size: usize, sample_rate: usize, result: &mut [T])
where
    T: Float,
{
    assert_eq!(size, result.len());

    let dh = T::from_usize(size).unwrap() / T::from_usize(sample_rate).unwrap();

    let max_freq = T::from_usize(size / 2).unwrap() / dh;
    let one = T::one();
    let two = one + one;

    let mut nodes = vec![];

    let mut f = freq;
    while f < max_freq {
        nodes.push(f);
        f = f * two;
    }

    for val in result.iter_mut() {
        *val = one;
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
