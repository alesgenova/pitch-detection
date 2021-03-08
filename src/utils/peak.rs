use crate::float::Float;

pub enum PeakCorrection {
    Quadratic,
    None,
}

/// An iterator that returns the positive peaks of `self.data`,
/// skipping over any initial positive values (i.e., every peak
/// is preceded by negative values).
struct PeaksIter<'a, T> {
    index: usize,
    data: &'a [T],
}

impl<'a, T: Float> PeaksIter<'a, T> {
    fn new(arr: &'a [T]) -> Self {
        Self {
            data: arr,
            index: 0,
        }
    }
}

impl<'a, T: Float> Iterator for PeaksIter<'a, T> {
    type Item = (usize, T);

    fn next(&mut self) -> Option<(usize, T)> {
        let mut idx = self.index;
        let mut max = -T::infinity();
        let mut max_index = idx;

        if idx >= self.data.len() {
            return None;
        }

        if idx == 0 {
            // If we're first starting iteration, we want to skip over
            // any positive values at the start of `self.data`. These are not
            // relevant for auto-correlation algorithms (since self.data[0] will always
            // be a global maximum for an autocorrelation).
            idx += self
                .data
                .iter()
                // `!val.is_sign_negative()` is used instead of `val.is_sign_positive()`
                // to make sure that any spurious NaN at the start are also skipped (NaN
                // is not sign positive and is not sign negative).
                .take_while(|val| !val.is_sign_negative())
                .count();
        }

        // Skip over the negative parts because we're looking for a positive peak!
        idx += self.data[idx..]
            .iter()
            .take_while(|val| val.is_sign_negative())
            .count();

        // Record the local max and max_index for the next stretch of positive values.
        for val in &self.data[idx..] {
            if val.is_sign_negative() {
                break;
            }
            if *val > max {
                max = *val;
                max_index = idx;
            }
            idx += 1;
        }

        self.index = idx;

        // We may not have found a maximum; the only time when this happens is when we've
        // reached the end of `self.data`. Alternatively, if `self.data` ends in a positive
        // segment we don't want to count `max` as a real maximum (since the data
        // was probably truncated in some way). In this case, we have `idx == self.data.len()`,
        // and so we terminate the iterator.
        if max == -T::infinity() || idx == self.data.len() {
            return None;
        }

        Some((max_index, max))
    }
}

// Find `(index, value)` of positive peaks in `arr`. Every positive peak is preceded and succeeded
// by negative values, so any initial positive segment of `arr` does not produce a peak.
pub fn detect_peaks<'a, T: Float>(arr: &'a [T]) -> impl Iterator<Item = (usize, T)> + 'a {
    PeaksIter::new(arr)
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

    #[test]
    fn detect_peaks_test() {
        let v = vec![-2., -1., 1., 2., 1., -1., 4., -3., -2., 1., 1., -1.];
        let peaks: Vec<(usize, f64)> = detect_peaks(v.as_slice()).collect();
        assert_eq!(peaks, [(3, 2.), (6, 4.), (9, 1.)]);

        let v = vec![1., 2., 1., -1., 2., -3., -2., 1., 1., -1.];
        let peaks: Vec<(usize, f64)> = detect_peaks(v.as_slice()).collect();
        assert_eq!(peaks, [(4, 2.), (7, 1.)]);

        let v = vec![1., 2., 1., -1., 2., -3., -2., 1., 1.];
        let peaks: Vec<(usize, f64)> = detect_peaks(v.as_slice()).collect();
        assert_eq!(peaks, [(4, 2.)]);
    }
}
