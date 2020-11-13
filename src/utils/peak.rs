use crate::float::Float;

pub enum PeakCorrection {
  Quadratic,
  None
}

struct Point<T: Float> {
  x: T,
  y: T
}

fn detect_crossings<T: Float>(arr: &[T]) -> Vec<(usize, usize)> {
  let mut crossings = Vec::new();
  let mut positive_zero_cross: Option<usize> = None;
  for i in 1..arr.len() {
      let val = arr[i];
      let prev_val = arr[i - 1];
      match positive_zero_cross {
          Some(idx) => {
              if val < T::zero() && prev_val > T::zero() {
                  crossings.push((idx, i));
                  positive_zero_cross = None;
              }
          },
          None => {
              if val > T::zero() && prev_val < T::zero() {
                  positive_zero_cross = Some(i);
              }
          }
      }
  }
  crossings
}

pub fn detect_peaks<T: Float>(arr: &[T]) -> Vec<(usize, T)> {
  let crossings = detect_crossings(arr);
  let mut peaks = Vec::new();

  for crossing in crossings {
      let (start, stop) = crossing;
      let mut peak_idx = 0;
      let mut peak_val = - T::infinity();
      for i in start..stop {
          if arr[i] > peak_val {
              peak_val = arr[i];
              peak_idx = i;
          }
      }
      peaks.push((peak_idx, peak_val));
  }

  peaks
}

pub fn choose_peak<T: Float>(peaks: &[(usize, T)], threshold: T) -> Option<(usize, T)> {
  let mut chosen: Option<(usize, T)> = None;
  for &peak in peaks {
      if peak.1 > threshold {
          chosen = Some(peak);
          break;
      }
  }
  chosen
}

pub fn correct_peak<T: Float>(peak: (usize, T), data: &[T], correction: PeakCorrection) -> (T, T) {
  match correction {
      PeakCorrection::Quadratic => {
          let idx = peak.0;
          let point = quadratic_interpolation(
              Point{x: T::from_usize(idx - 1).unwrap(), y: data[idx - 1]},
              Point{x: T::from_usize(idx).unwrap(), y: data[idx]},
              Point{x: T::from_usize(idx + 1).unwrap(), y: data[idx + 1]},
          );
          return (point.x, point.y);
      },
      PeakCorrection::None => {
          return (T::from_usize(peak.0).unwrap(), peak.1);
      }
  }
}

fn quadratic_interpolation<T:Float>(left: Point<T>, center: Point<T>, right: Point<T>) -> Point<T> {
  let shift = T::from_f64(0.5).unwrap() * (right.y - left.y) / (T::from_f64(2.0).unwrap() * center.y - left.y - right.y);
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
            Point{x: -1.5, y: - (1.5 * 1.5) + 4.0},
            Point{x: -0.5, y: - (0.5 * 0.5) + 4.0},
            Point{x: 0.5, y: - (0.5 * 0.5) + 4.0}
        );
        assert_eq!(point.x, 0.0);
        assert_eq!(point.y, 4.0);
    }
}