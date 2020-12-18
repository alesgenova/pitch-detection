use crate::float::Float;

pub enum PeakCorrection {
    Quadratic,
    None,
}

struct Point<T: Float> {
    x: T,
    y: T,
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
            let point = quadratic_interpolation(
                Point {
                    x: T::from_usize(idx - 1).unwrap(),
                    y: data[idx - 1],
                },
                Point {
                    x: T::from_usize(idx).unwrap(),
                    y: data[idx],
                },
                Point {
                    x: T::from_usize(idx + 1).unwrap(),
                    y: data[idx + 1],
                },
            );
            return (point.x, point.y);
        }
        PeakCorrection::None => {
            return (T::from_usize(peak.0).unwrap(), peak.1);
        }
    }
}

fn quadratic_interpolation<T: Float>(
    left: Point<T>,
    center: Point<T>,
    right: Point<T>,
) -> Point<T> {
    let shift = T::from_f64(0.5).unwrap() * (right.y - left.y)
        / (T::from_f64(2.0).unwrap() * center.y - left.y - right.y);
    let x = center.x + shift;
    let y = center.y + T::from_f64(0.25).unwrap() * (right.y - left.y) * shift;
    Point { x, y }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_correction() {
        let point = quadratic_interpolation(
            Point {
                x: -1.5,
                y: -(1.5 * 1.5) + 4.0,
            },
            Point {
                x: -0.5,
                y: -(0.5 * 0.5) + 4.0,
            },
            Point {
                x: 0.5,
                y: -(0.5 * 0.5) + 4.0,
            },
        );
        assert_eq!(point.x, 0.0);
        assert_eq!(point.y, 4.0);
    }
}
