//! Ordinary least squares with an intercept, for the replay bench's cost model
//! (`examples/replay --model`): a pass's cycles against what the pass did.

/// `y ≈ intercept + Σ coef[j] · x[j]` over the rows it was fitted on.
#[derive(Clone, Debug)]
pub struct Fit {
    pub rows: usize,
    pub intercept: f64,
    pub intercept_se: f64,
    /// Per column, its coefficient and standard error; a column that never varies gets 0
    /// and NaN, and its constant value stays in the intercept.
    pub coef: Vec<f64>,
    pub se: Vec<f64>,
    pub mean_x: Vec<f64>,
    pub mean_y: f64,
    /// The share of `y`'s variance the columns explain, and the residuals' standard deviation.
    pub r2: f64,
    pub residual_sd: f64,
}

impl Fit {
    pub fn predict(&self, x: &[f64]) -> f64 {
        self.intercept + self.coef.iter().zip(x).map(|(c, v)| c * v).sum::<f64>()
    }
}

/// Fits `rows` rows of `k` columns, `x(row, column)` against `y(row)`, by the normal
/// equations over the centred columns. Err when two varying columns are collinear.
pub fn least_squares(rows: usize, k: usize, x: impl Fn(usize, usize) -> f64, y: impl Fn(usize) -> f64) -> Result<Fit, String> {
    if rows < k + 2 {
        return Err(format!("{rows} rows for {k} columns"));
    }
    let n = rows as f64;
    let mean_x: Vec<f64> = (0..k).map(|j| (0..rows).map(|i| x(i, j)).sum::<f64>() / n).collect();
    let mean_y = (0..rows).map(&y).sum::<f64>() / n;
    // The centred sums of squares and products.
    let mut xx = vec![0f64; k * k];
    let mut xy = vec![0f64; k];
    let mut yy = 0f64;
    let mut row = vec![0f64; k];
    for i in 0..rows {
        for j in 0..k {
            row[j] = x(i, j) - mean_x[j];
        }
        let dy = y(i) - mean_y;
        yy += dy * dy;
        for a in 0..k {
            xy[a] += row[a] * dy;
            for b in a..k {
                xx[a * k + b] += row[a] * row[b];
            }
        }
    }
    for a in 0..k {
        for b in 0..a {
            xx[a * k + b] = xx[b * k + a];
        }
    }
    let live: Vec<usize> = (0..k).filter(|&j| xx[j * k + j] > 0.0).collect();
    let m = live.len();
    // Gauss-Jordan on [A | I] with partial pivoting, the columns scaled to unit variance
    // so that one tolerance fits them all.
    let scale: Vec<f64> = live.iter().map(|&j| xx[j * k + j].sqrt()).collect();
    let mut a = vec![0f64; m * 2 * m];
    for (r, &jr) in live.iter().enumerate() {
        for (c, &jc) in live.iter().enumerate() {
            a[r * 2 * m + c] = xx[jr * k + jc] / (scale[r] * scale[c]);
        }
        a[r * 2 * m + m + r] = 1.0;
    }
    for c in 0..m {
        let p = (c..m).max_by(|&p, &q| a[p * 2 * m + c].abs().total_cmp(&a[q * 2 * m + c].abs())).expect("a row");
        if a[p * 2 * m + c].abs() < 1e-9 {
            return Err(format!("column {} is a combination of the columns before it", live[c]));
        }
        for t in 0..2 * m {
            a.swap(c * 2 * m + t, p * 2 * m + t);
        }
        let d = a[c * 2 * m + c];
        for t in 0..2 * m {
            a[c * 2 * m + t] /= d;
        }
        for r in (0..m).filter(|&r| r != c) {
            let f = a[r * 2 * m + c];
            if f != 0.0 {
                for t in 0..2 * m {
                    a[r * 2 * m + t] -= f * a[c * 2 * m + t];
                }
            }
        }
    }
    let inv = |r: usize, c: usize| a[r * 2 * m + m + c] / (scale[r] * scale[c]);
    let beta: Vec<f64> = (0..m).map(|r| (0..m).map(|c| inv(r, c) * xy[live[c]]).sum()).collect();
    let explained: f64 = (0..m).map(|r| beta[r] * xy[live[r]]).sum();
    let rss = (yy - explained).max(0.0);
    let var = rss / (n - m as f64 - 1.0);
    let (mut coef, mut se) = (vec![0f64; k], vec![f64::NAN; k]);
    for (r, &j) in live.iter().enumerate() {
        coef[j] = beta[r];
        se[j] = (var * inv(r, r)).sqrt();
    }
    let intercept = mean_y - (0..k).map(|j| coef[j] * mean_x[j]).sum::<f64>();
    let quad: f64 = (0..m).map(|r| (0..m).map(|c| mean_x[live[r]] * inv(r, c) * mean_x[live[c]]).sum::<f64>()).sum();
    Ok(Fit { rows, intercept, intercept_se: (var * (1.0 / n + quad)).sqrt(), coef, se, mean_x, mean_y, r2: if yy > 0.0 { explained / yy } else { 1.0 }, residual_sd: var.sqrt() })
}

#[cfg(test)]
mod tests {
    use super::least_squares;

    /// A fixed pseudo-random stream (an LCG's high bits), uniform in [0, 1).
    fn stream(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn recovers_known_coefficients_through_noise() {
        let mut next = stream(7);
        // Three columns, the second correlated with the first, the fourth constant.
        let rows: Vec<[f64; 5]> = (0..20_000)
            .map(|_| {
                let a = (next() * 900.0).floor();
                let b = (a / 5.0 + next() * 40.0).floor();
                let c = (next() * 4.0).floor();
                let noise = (next() + next() + next() - 1.5) * 200.0;
                [a, b, c, 3.0, 2500.0 + 6.5 * a + 14.0 * b + 210.0 * c + noise]
            })
            .collect();
        let fit = least_squares(rows.len(), 4, |i, j| rows[i][j], |i| rows[i][4]).unwrap();
        for (got, want, tol) in [(fit.intercept, 2500.0, 12.0), (fit.coef[0], 6.5, 0.05), (fit.coef[1], 14.0, 0.3), (fit.coef[2], 210.0, 3.0)] {
            assert!((got - want).abs() < tol, "{got} against {want}: {fit:?}");
        }
        assert_eq!(fit.coef[3], 0.0);
        assert!(fit.se[3].is_nan());
        // The noise is a sum of three uniforms over 200: sd 200 · sqrt(3 / 12) = 100.
        assert!((fit.residual_sd - 100.0).abs() < 3.0, "{}", fit.residual_sd);
        assert!(fit.r2 > 0.99 && (fit.predict(&[100.0, 30.0, 2.0, 3.0]) - 3990.0).abs() < 12.0);
        // Each coefficient lies within four standard errors of the truth.
        assert!((fit.coef[0] - 6.5).abs() < 4.0 * fit.se[0] && (fit.intercept - 2500.0).abs() < 4.0 * fit.intercept_se);
    }

    #[test]
    fn refuses_collinear_columns() {
        let err = least_squares(100, 2, |i, j| (i * (j + 1)) as f64, |i| i as f64).unwrap_err();
        assert_eq!(err, "column 1 is a combination of the columns before it");
    }
}
