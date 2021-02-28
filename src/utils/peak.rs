use crate::float::Float;

pub enum PeakCorrection {
    Quadratic,
    None,
}

fn detect_crossings<'a, T: Float>(arr: &'a [T]) -> impl Iterator<Item = (usize, usize)> + 'a {
    arr.windows(2)
        .enumerate()
        .scan(
            None,
            |positive_zero_cross: &mut Option<usize>, (i, win)| match positive_zero_cross.take() {
                Some(idx) => {
                    if win[1] < T::zero() && win[0] > T::zero() {
                        *positive_zero_cross = None;
                        Some(Some((idx, i + 1)))
                    } else {
                        *positive_zero_cross = Some(idx);
                        Some(None)
                    }
                }
                None => {
                    if win[1] > T::zero() && win[0] < T::zero() {
                        *positive_zero_cross = Some(i + 1);
                    }
                    Some(None)
                }
            },
        )
        .filter_map(|o| o)
}

pub fn detect_peaks<'a, T: Float>(arr: &'a [T]) -> impl Iterator<Item = (usize, T)> + 'a {
    detect_crossings(arr).map(move |(start, stop)| {
        let mut peak_idx = 0;
        let mut peak_val = -T::infinity();
        for i in start..stop {
            if arr[i] > peak_val {
                peak_val = arr[i];
                peak_idx = i;
            }
        }
        (peak_idx, peak_val)
    })
}

pub fn choose_peak<I: Iterator<Item = (usize, T)>, T: Float>(
    mut peaks: I,
    threshold: T,
) -> Option<(usize, T)> {
    peaks.find(|p| p.1 > threshold)
}

pub fn correct_peak<T: Float>(peak: (usize, T), data: &[T], correction: PeakCorrection) -> (T, T) {
    match correction {
        PeakCorrection::Quadratic => {
            let idx = peak.0;
            let (x, y) = find_quadratic_peak(data[idx - 1], data[idx], data[idx + 1]);
            (x + T::from_usize(idx).unwrap(), y)
        }
        PeakCorrection::None => (T::from_usize(peak.0).unwrap(), peak.1),
    }
}

/// Use a quadratic interpolation to find the maximum of
/// a parabola passing through `(-1, y0)`, `(0, y1)`, `(1, y2)`.
///
/// The output is of the form `(x-offset, peak value)`.
fn find_quadratic_peak<T: Float>(y0: T, y1: T, y2: T) -> (T, T) {
    // The quadratic ax^2+bx+c passing through
    // (-1, y0), (0, y1), (1, y2), the
    // has coefficients
    //
    // a = y0/2 - y1 + y2/2
    // b = (y2 - y0)/2
    // c = y1
    //
    // and a maximum at x=-b/(2a) and y=-b^2/(4a) + c

    // Some constants
    let two = T::from_f64(2.).unwrap();
    let four = T::from_f64(4.).unwrap();

    let a = (y0 + y2) / two - y1;
    let b = (y2 - y0) / two;
    let c = y1;

    // If we're concave up, the maximum is at one of the end points
    if a > T::zero() {
        if y0 > y2 {
            return (-T::one(), y0);
        }
        return (T::one(), y2);
    }

    (-b / (two * a), -b * b / (four * a) + c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_correction() {
        fn quad1(x: f64) -> f64 {
            -x * x + 4.0
        }
        let (x, y) = find_quadratic_peak(quad1(-1.5), quad1(-0.5), quad1(0.5));
        assert_eq!(x - 0.5, 0.0);
        assert_eq!(y, 4.0);
        let (x, y) = find_quadratic_peak(quad1(-3.5), quad1(-2.5), quad1(-1.5));
        assert_eq!(x - 2.5, 0.0);
        assert_eq!(y, 4.0);

        fn quad2(x: f64) -> f64 {
            -2. * x * x + 3. * x - 2.5
        }

        let (x, y) = find_quadratic_peak(quad2(-1.), quad2(0.), quad2(1.));
        assert_eq!(x, 0.75);
        assert_eq!(y, -1.375);

        // Test of concave-up parabolas
        fn quad3(x: f64) -> f64 {
            x * x + 2.0
        }

        let (x, y) = find_quadratic_peak(quad3(0.), quad3(1.), quad3(2.));
        assert_eq!(x + 1., 2.);
        assert_eq!(y, 6.);
        let (x, y) = find_quadratic_peak(quad3(-2.), quad3(-1.), quad3(0.));
        assert_eq!(x - 1., -2.);
        assert_eq!(y, 6.);
    }
}
