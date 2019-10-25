// use std::ops::Index;
use std::ops::Mul;
use std::ops::Add;
use std::iter::Sum;
use std::ops::Sub;
use std::ops::Div;
use std::ops::AddAssign;
use std::cmp::PartialOrd;
// use std::cmp::PartialEq;

#[cfg(test)]
mod tests {

    #[test]
    fn test_array_addition() {
        use crate::array_addition;

        assert_eq!(array_addition(&[1, 2], &[2, 1]), &[3, 3]);
    }

    #[test]
    fn test_array_multiplication_int() {
        use crate::array_multiplication;

        assert_eq!(array_multiplication(&[2, 2], &[2, 1]), &[4, 2]);
    }

    #[test]
    fn test_array_multiplication_float() {
        use crate::array_multiplication;

        assert_eq!(array_multiplication(&[2.1, 2.0], &[2.0, 1.1]), &[4.2, 2.2]);
    }

    #[test]
    fn test_matrix_multiplication_samesize() {
        use crate::matrix_multiplication;
        use crate::DataFrame;

        let a = DataFrame {
            rows: 2,
            cols: 2,
            data: [1, 2,
                   0, 1
                   ].to_vec(),
        };

        let b = DataFrame {
            rows: 2,
            cols: 2,
            data: [2, 1,
                   1, 0
                   ].to_vec(),
        };

        let ab = DataFrame {
            rows: 2,
            cols: 2,
            data: [4, 1,
                   1, 0
                   ].to_vec(),
        };

        assert_eq!(matrix_multiplication(&a, &b), ab);
    }

    #[test]
    fn test_matrix_multiplication_differentsize() {
        use crate::matrix_multiplication;
        use crate::DataFrame;

        let a = DataFrame {
            rows: 2,
            cols: 3,
            data: [1, 2, 1,
                   0, 1, 1
                   ].to_vec(),
        };

        let b = DataFrame {
            rows: 3,
            cols: 2,
            data: [2, 1,
                   1, 0,
                   3, 2
                   ].to_vec(),
        };

        let ab = DataFrame {
            rows: 2,
            cols: 2,
            data: [7, 3,
                   4, 2
                   ].to_vec(),
        };

        assert_eq!(matrix_multiplication(&a, &b), ab);
    }

    #[test]
    #[ignore]
    fn test_matrix_inversion() {
        use crate::DataFrame;

        let a = DataFrame {
            rows: 2,
            cols: 2,
            data: [1, 2,
                   0, 1
                   ].to_vec(),
        };

        let a_inverted = DataFrame {
            rows: 2,
            cols: 2,
            data: [].to_vec(),
        };

        assert_eq!(a.invert(), a_inverted);
    }

    #[test]
    fn test_matrix_transposition_square() {
        use crate::DataFrame;

        let a = DataFrame {
            rows: 2,
            cols: 2,
            data: [1, 2,
                   0, 1
                   ].to_vec(),
        };

        let a_transposed = DataFrame {
            rows: 2,
            cols: 2,
            data: [1, 0,
                   2, 1
                   ].to_vec(),
        };

        assert_eq!(a.transpose(), a_transposed);
    }

    #[test]
    fn test_matrix_multiplication_scalar() {
        use crate::DataFrame;

        let a = DataFrame {
            rows: 2,
            cols: 2,
            data: [1.0, 2.0,
                   0.0, 1.0
                   ].to_vec(),
        };

        let a_multiplied = DataFrame {
            rows: 2,
            cols: 2,
            data: [2.0, 4.0,
                   0.0, 2.0
                   ].to_vec(),
        };

        assert_eq!(2.0 * &a, a_multiplied);
    }

    #[test]
    fn test_vector_len() {
        use crate::DataFrame;

        let a = DataFrame {
            rows: 2,
            cols: 1,
            data: [1.0,
                   2.0,
                   ].to_vec(),
        };

        let a_len: f64 = 5.0_f64.sqrt();

        assert_eq!(a.len(), a_len);
    }

    #[test]
    fn test_matrix_pop() {
        use crate::DataFrame;

        let a = DataFrame {
            rows: 2,
            cols: 1,
            data: [1.0,
                   2.0,
                   ].to_vec(),
        };

        let a_popped = DataFrame {
            rows: 1,
            cols: 1,
            data: [1.0].to_vec(),
        };

        assert_eq!(a.pop(), a_popped);
    }

    #[test]
    fn test_matrix_subtraction() {
        use crate::DataFrame;

        let a = DataFrame {
            rows: 2,
            cols: 2,
            data: [1.0, 2.0,
                   1.0, 1.0
                   ].to_vec(),
        };

        let b = DataFrame {
            rows: 2,
            cols: 2,
            data: [3.0, 2.0,
                   0.0, 7.0
                   ].to_vec(),
        };

        let b_a = DataFrame {
            rows: 2,
            cols: 2,
            data: [2.0, 0.0,
                   -1.0, 6.0
                   ].to_vec(),
        };

        assert_eq!(&b - &a, b_a);
    }

    #[test]
    fn test_matrix_transposition_rect() {
        use crate::DataFrame;

        let a = DataFrame {
            rows: 2,
            cols: 3,
            data: [1, 2, 0,
                   1, 3, 2
                   ].to_vec(),
        };

        let a_transposed = DataFrame {
            rows: 3,
            cols: 2,
            data: [1, 1,
                   2, 3,
                   0, 2
                   ].to_vec(),
        };

        assert_eq!(a.transpose(), a_transposed);
    }

    #[test]
    fn test_matrix_lu_decomposition_small() {
        use crate::DataFrame;
        use crate::lu_decompose;

        let a = DataFrame {
            rows: 2,
            cols: 2,
            data: [3, 1,
                   4, 2
                   ].to_vec(),
        };

        let lu = DataFrame {
            rows: 2,
            cols: 2,
            data: [3_f64, 1.0,
                   1.3333333333333333, 0.6666666666666667
                  ].to_vec(),
        };

        assert_eq!(lu_decompose(&a), lu);
    }

    #[test]
    fn test_matrix_lu_decomposition_large() {
        use crate::DataFrame;
        use crate::lu_decompose;

        let b = DataFrame {
            rows: 3,
            cols: 3,
            data: [1, 2, 3,
                   4, 5, 6,
                   7, 8, 9
                   ].to_vec(),
        };

        let lu = DataFrame {
            rows: 3,
            cols: 3,
            data: [1_f64, 2.0, 3.0,
                   4.0, -3.0, -6.0,
                   7.0, 2.0, 0.0
                   ].to_vec(),
        };

        assert_eq!(lu_decompose(&b), lu);
    }

    #[test]
    fn test_matrix_append_col() {
        use crate::DataFrame;

        let a = DataFrame {
            rows: 3,
            cols: 3,
            data: [1, 2, 3,
                   4, 5, 6,
                   7, 8, 9
                   ].to_vec(),
        };

        let other = DataFrame {
            rows: 3,
            cols: 1,
            data: [1,
                   1,
                   1,
                   ].to_vec(),
        };

        let appended = DataFrame {
            rows: 3,
            cols: 4,
            data: [1, 2, 3, 1,
                   4, 5, 6, 1,
                   7, 8, 9, 1,
                   ].to_vec(),
        };

        assert_eq!(a.append_col(&other), appended);
    }

    #[test]
    fn test_linear_regression_small_lu() {
        use crate::DataFrame;
        use crate::linear_regression_lu;

        let data = DataFrame {
            rows: 3,
            cols: 3,
            data: [1, 2, 3,
                   2, 3, 4,
                   2, 4, 5,
                   ].to_vec(),
        };

        let target = DataFrame {
            rows: 3,
            cols: 1,
            data: [14,
                   20,
                   25,
                   ].to_vec(),
        };

        let regression = DataFrame {
            rows: 3,
            cols: 1,
            data: [1_f64,
                   2.0,
                   3.0,
                   ].to_vec(),
        };

        assert_eq!(linear_regression_lu(&data, &target), regression);
    }

    #[test]
    fn test_linear_regression_small_gd() {
        use crate::DataFrame;
        use crate::linear_regression_gd;
        use assert_approx_eq::assert_approx_eq;

        let data = DataFrame {
            rows: 4,
            cols: 3,
            data: [1.0, 2.0, 3.0,
                   2.0, 3.0, 4.0,
                   2.0, 4.0, 5.0,
                   1.0, 2.0, 2.5,
                   ].to_vec(),
        };

        let target = DataFrame {
            rows: 4,
            cols: 1,
            data: [15.0,
                   21.0,
                   26.0,
                   13.5,
                   ].to_vec(),
        };

        let regression = DataFrame {
            rows: 3,
            cols: 1,
            data: [1_f64,
                   2.0,
                   3.0,
                   ].to_vec(),
        };

        let b: f64 = 1.0;

        let result = linear_regression_gd(&data, &target, 0.0001, 0.001, 100000);

        assert_approx_eq!(&result.0, &regression, 0.001);
        assert_approx_eq!(&result.1, &b, 0.001);
    }

    #[test]
    fn test_linear_regression_medium_gd() {
        use crate::DataFrame;
        use crate::linear_regression_gd;
        use assert_approx_eq::assert_approx_eq;

        let data = DataFrame {
            rows: 3,
            cols: 3,
            data: [1.0, 2.0, 3.0,
                   2.0, 3.0, 4.0,
                   2.0, 4.0, 5.0,
                   2.2, 4.0, 5.3,
                   8.3, 5.4, 7.2,
                   1.2, 5.3, 4.3,
                   1.5, 0.3, 6.3,
                   2.15, 3.7, 9.2,
                   1.05, 2.3, 4.4,
                   4.0, 4.0, 2.0,
                   ].to_vec(),
        };

        let target = DataFrame {
            rows: 3,
            cols: 1,
            data: [14.0,
                   20.7,
                   25.2,
                   26.0,
                   40.6,
                   25.0,
                   22.3,
                   35.7,
                   17.3,
                   19.4,
                   ].to_vec(),
        };

        let regression = DataFrame {
            rows: 3,
            cols: 1,
            data: [2.2_f64,
                   2.4,
                   2.1,
                   ].to_vec(),
        };

        let b: f64 = 0.7;

        let result = linear_regression_gd(&data, &target, 0.0001, 0.001, 100000);

        assert_approx_eq!(&result.0, &regression, 0.001);
        assert_approx_eq!(&result.1, &b, 0.001);
    }

    #[test]
    fn test_ridge_regression_small() {
        use crate::DataFrame;
        use crate::ridge_regression;
        use assert_approx_eq::assert_approx_eq;

        let data = DataFrame {
            rows: 4,
            cols: 3,
            data: [1.0, 2.0, 3.0,
                   2.0, 3.0, 4.0,
                   2.0, 4.0, 5.0,
                   1.0, 2.0, 2.5,
                   ].to_vec(),
        };

        let target = DataFrame {
            rows: 4,
            cols: 1,
            data: [15.0,
                   21.0,
                   26.0,
                   13.5,
                   ].to_vec(),
        };

        let regression = DataFrame {
            rows: 3,
            cols: 1,
            data: [1_f64,
                   2.0,
                   3.0,
                   ].to_vec(),
        };

        let b: f64 = 1.0;

        let result = ridge_regression(&data, &target, 0.0001, 0.001, 10000000, 1.0);

        assert_approx_eq!(&result.0, &regression, 0.001);
        assert_approx_eq!(&result.1, &b, 0.001);
    }
}


pub fn array_addition(a: &[i32], b:&[i32]) -> Vec<i32> {
    assert_eq!(a.len(), b.len(), "The vectors are not the same length.");
    let mut c: Vec<i32> = vec![0; b.len()];
    for item in 0..a.len() {
        c[item] = a[item] + b[item];
    }
    c
}

pub fn array_multiplication<T: Mul<Output = T> + Copy>(a: &[T], b:&[T]) -> Vec<T> {
    assert_eq!(a.len(), b.len(), "The vectors are not the same length.");
    let mut c: Vec<T> = vec![];
    for item in 0..a.len() {
        c.push(a[item] * b[item]);
    }
    c
}

pub fn matrix_multiplication<'t, T: 't + Mul<Output = T> + Add<Output = T> + Copy + Sum>(a: &DataFrame<T>, b: &DataFrame<T>) -> DataFrame<T> {
    assert_eq!(a.cols, b.rows, "The matrices are incompatible.");
    let mut c = DataFrame {
        rows: a.rows,
        cols: b.cols,
        data: Vec::new(),
    };

    for a_item in 0..a.rows {
        for b_item in 0..b.cols {
            c.data.push(array_multiplication(&a.row(a_item), &b.col(b_item)).iter().copied().sum());
        }
    }
    c
}

#[derive(Debug, PartialEq)]
pub struct DataFrame<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Copy> DataFrame<T> {
    fn get(&self, row: usize, col: usize) -> T {
        assert!(row < self.rows);
        assert!(col < self.cols);
        self.data[row * self.cols + col]
    }
}

impl<T: Copy> DataFrame<T> {
    fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.rows);
        assert!(col < self.cols);
        self.data[row * self.cols + col] = value;
    }
}

impl<T: Copy> DataFrame<T> {
    fn col(&self, col: usize) -> Vec<T> {
        assert!(col < self.cols);
        let mut data = vec![];
        for row in 0..self.rows {
            data.push(self.get(row, col));
        }
        data
    }
}

impl<T: Copy> DataFrame<T> {
    fn row(&self, row: usize) -> &[T] {
        assert!(row < self.rows);
        let idx_start = row * self.cols;
        let idx_end = idx_start + self.cols;
        &self.data[idx_start..idx_end]
    }
}

impl<T: Copy + Mul<Output = T> + Into<f64> + From<f64> + Add<Output = T>> DataFrame<T> {
    fn len(&self) -> T {
        assert!(self.cols == 1);
        let vec_len: f64 = self.data.iter().fold(0.0, |acc, &x| acc + x.into() * x.into());
        T::from(vec_len.sqrt())
    }
}

impl<T: Copy + Mul<Output = T> + Into<f64> + From<f64> + Add<Output = T>> DataFrame<T> {
    fn abs(&self) -> T {
        self.len()
    }
}

impl<T: Copy> DataFrame<T> {
    fn pop(&self) -> DataFrame<T> {
        DataFrame {
            rows: self.rows - 1,
            cols: self.cols,
            data: self.data[0.. (self.rows - 1) * self.cols].to_vec(),
        }
    }
}

impl<T: Copy> DataFrame<T> {
    fn invert(&self) -> DataFrame<T> {
        //cholesky = get_cholesky(self);

        DataFrame {
            rows: self.rows,
            cols: self.cols,
            data: (&*self.data).to_vec(),
        }
    }
}

impl<T: Copy> DataFrame<T> {
    fn transpose(&self) -> DataFrame<T> {
        let mut data = vec![];
        for col in 0..self.cols {
            data.extend(self.col(col));
        }

        DataFrame {
            rows: self.cols,
            cols: self.rows,
            data: data,
        }
    }
}

impl<T: Copy> DataFrame<T> {
    fn append_col(&self, other: &DataFrame<T>) -> DataFrame<T> {
        assert!(other.cols == 1);
        let mut data = vec![];
        for row in 0..self.rows {
            data.extend(self.row(row));
            data.extend(other.row(row));
        }

        DataFrame {
            rows: self.rows,
            cols: self.cols + 1,
            data: data,
        }
    }
}

impl<T: Mul<Output = T> + Copy + From<f64>> std::ops::Mul<&DataFrame<T>> for f64 {
    type Output = DataFrame<T>;

    fn mul(self, right: &DataFrame<T>) -> DataFrame<T> {
        DataFrame {
            rows: right.rows,
            cols: right.cols,
            data: right.data.iter().map(|&x| (T::from(self)) * x).collect(),
        }
    }
}

impl<T: Sub<Output = T> + Copy + From<f64>> std::ops::Sub<&DataFrame<T>> for &DataFrame<T> {
    type Output = DataFrame<T>;

    fn sub(self, right: &DataFrame<T>) -> DataFrame<T> {
        DataFrame {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().zip(right.data.iter()).map(|(&x, &y)| x - y).collect(),
        }
    }
}

impl<T: Add<Output = T> + Copy + From<f64>> std::ops::Add<&DataFrame<T>> for &DataFrame<T> {
    type Output = DataFrame<T>;

    fn add(self, right: &DataFrame<T>) -> DataFrame<T> {
        DataFrame {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().zip(right.data.iter()).map(|(&x, &y)| x + y).collect(),
        }
    }
}

pub fn lu_decompose<T: Copy + From<u8> + Into<f64> + Sub<Output = T> + Div<Output = T> + AddAssign + Mul<Output = T>> (matrix: &DataFrame<T>) -> DataFrame<f64> {
    let mut lu = DataFrame {
        rows: matrix.rows,
        cols: matrix.cols,
        data: vec![0_f64; matrix.rows * matrix.cols],
    };
    let mut sum: f64;

    for i in 0..matrix.rows {
        for j in i..matrix.rows {
            sum = 0_f64;
            for k in 0..i {
                sum += lu.get(i, k) * lu.get(k, j);
            }
            lu.set(i, j, matrix.get(i, j).into() - sum);
        }
        for j in i+1..matrix.rows {
            sum = 0_f64;
            for k in 0..i {
                sum += lu.get(j, k) * lu.get(k, i);
            }
            lu.set(j, i, (matrix.get(j, i).into() - sum) / lu.get(i, i));
        }
    }
    lu
}

pub fn linear_regression_gd<T> (datapoints: &DataFrame<T>,
                                target: &DataFrame<T>,
                                tol: f64,
                                step_size: f64,
                                n_iter: i32) -> (DataFrame<T>, T) where
    T: Copy + Sub<Output = T> + PartialOrd + Sum + Add<Output = T> + Into<f64> + From<f64> + Mul<Output = T> {
    linear_regression_gd_normed(datapoints, target, tol, step_size, n_iter, 0.0, 0.0)
}

pub fn lasso_regression<T> (datapoints: &DataFrame<T>,
                            target: &DataFrame<T>,
                            tol: f64,
                            step_size: f64,
                            n_iter: i32,
                            alpha: f64) -> (DataFrame<T>, T) where
    T: Copy + Sub<Output = T> + PartialOrd + Sum + Add<Output = T> + Into<f64> + From<f64> + Mul<Output = T> {
    linear_regression_gd_normed(datapoints, target, tol, step_size, n_iter, alpha, 0.0)
}

pub fn ridge_regression<T> (datapoints: &DataFrame<T>,
                            target: &DataFrame<T>,
                            tol: f64,
                            step_size: f64,
                            n_iter: i32,
                            alpha: f64) -> (DataFrame<T>, T) where
    T: Copy + Sub<Output = T> + PartialOrd + Sum + Add<Output = T> + Into<f64> + From<f64> + Mul<Output = T> {
    linear_regression_gd_normed(datapoints, target, tol, step_size, n_iter, 0.0, alpha)
}

pub fn elastic_net_seq<T> (datapoints: &DataFrame<T>,
                       target: &DataFrame<T>,
                       tol: f64,
                       step_size: f64,
                       n_iter: i32,
                       alpha: f64) -> (DataFrame<T>, T) where
    T: Copy + Sub<Output = T> + PartialOrd + Sum + Add<Output = T> + Into<f64> + From<f64> + Mul<Output = T> {
    linear_regression_gd_normed(datapoints, target, tol, step_size, n_iter, 0.0, 0.0)
}

pub fn linear_regression_gd_normed<T> (datapoints: &DataFrame<T>,
                                       target: &DataFrame<T>,
                                       tol: f64,
                                       step_size: f64,
                                       n_iter: i32,
                                       l1_alpha: f64,
                                       l2_alpha: f64) -> (DataFrame<T>, T) where
    T: Copy + Sub<Output = T> + PartialOrd + Sum + Add<Output = T> + Into<f64> + From<f64> + Mul<Output = T> {
    // 2A^T * A * x - 2 A^T * y + l2_alpha * 2 * x + l1_alpha * signs(x) = grad

    let b_helper_col = DataFrame {
        rows: datapoints.rows,
        cols: 1,
        data: vec![T::from(1.0); datapoints.rows],
    };
    let data = datapoints.append_col(&b_helper_col);
    let mut grad: DataFrame<T> = DataFrame {
        rows: data.cols,
        cols: 1,
        data: vec![T::from(1.0); data.cols],
    };
    let mut x: DataFrame<T> = DataFrame {
        rows: data.cols,
        cols: 1,
        data: vec![T::from(1.0); data.cols],
    };

    let mut iter: i32 = 0;

    while (grad.len() > tol.into()) & (iter <= n_iter) {
        // calculate gradient

        let atax: DataFrame<T> = 2.0 * &matrix_multiplication(&matrix_multiplication(&data.transpose(), &data), &x);
        let aty: DataFrame<T> = 2.0 * &matrix_multiplication(&data.transpose(), &target);
        let l2_reg: DataFrame<T> = 2.0 * l2_alpha * &x;
        let l1_reg: DataFrame<T> = 2.0 * l1_alpha * &x;

        grad = &(&atax - &aty) + &(&l2_reg + &l1_reg);

        // update x
        x = &x - &(step_size * &grad);

        // update iter count
        iter += 1;
    }
    let b: T = x.row(x.rows - 1)[0];
    (x.pop(), b)
}


pub fn linear_regression_lu<T: Copy + From<u8> + Div<Output = T> + Mul<Output = T> + Into<f64> + AddAssign + Sub<Output = T>> (datapoints: &DataFrame<T>, target: &DataFrame<T>) -> DataFrame<f64> {
    // based on LU decomposition
    // theoretically also possible: via matrix transposition/inversion, however, computationally expensive/complicated
    // https://medium.com/@andrew.chamberlain/the-linear-algebra-view-of-least-squares-regression-f67044b7f39b

    let lu = lu_decompose(&datapoints);

    let mut y: Vec<f64> = vec![];
    let mut sum: f64;

    for i in 0..datapoints.rows {
        sum = 0.0;
        for k in 0..i {
            sum += lu.get(i, k) * y[k];
        }
        y.push(target.get(i, 0).into() - sum);
    }

    let mut x: Vec<f64> = vec![0.0; datapoints.cols];

    for i in (0..datapoints.rows).rev() {
        sum = 0.0;
        for k in (i+1)..datapoints.rows {
            sum += lu.get(i, k) * x[k];
        }
        x[i] = (y[i] - sum) / lu.get(i, i);
    }

    DataFrame {
        rows: datapoints.rows,
        cols: 1,
        data: x,
    }
}

/*impl<T> Index<usize> for DataFrame<T> {
    type Output = [T];

    fn index(&self, idx: usize) -> &Self::Output {
        assert!(idx[0] < self.rows);
        // assert!(col < self.cols);
        &self.data[0..2]
    }
}*/



/*impl<T> Index<usize> for DataFrame<T> {
    type Output = [T];

    fn index(&self, idx: [usize]) -> &Self::Output {
        assert!(idx[0] < self.rows);
        assert!(idx[1] < self.cols);
        let mut data = vec![];

        for row_idx in idx[0] {
            for col_idx in idx[1] {
                data.push(self.get(row_idx, col_idx));
            }
        }

        &data
    }
}*/

/*
impl PartialEq<DataFrame> for DataFrame<usize> {
    fn eq(&self, other: &DataFrame) -> bool {
        // this should work
        self.cols == other.cols and self.rows == other.rows and self.data == other.data
    }
}
*/
