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

        let data = DataFrame {
            rows: 3,
            cols: 3,
            data: [1.0, 2.0, 3.0,
                   2.0, 3.0, 4.0,
                   2.0, 4.0, 5.0,
                   ].to_vec(),
        };

        let target = DataFrame {
            rows: 3,
            cols: 1,
            data: [14.0,
                   20.0,
                   25.0,
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

        assert_eq!(linear_regression_gd(&data, &target), regression);
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
            data.extend(self.col(col))
        }

        DataFrame {
            rows: self.cols,
            cols: self.rows,
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

pub fn linear_regression_gd<T: Copy + Sub<Output = T> + PartialOrd + Sum + Add<Output = T> + Into<f64> + From<f64> + Mul<Output = T>> (datapoints: &DataFrame<T>, target: &DataFrame<T>) -> DataFrame<T> {
    // 2A^T * A * x - 2 A^T * y = grad

    let tol: f64 = 0.0001;
    let step_size: f64 = 0.01;
    let mut grad: DataFrame<T> = DataFrame {
        rows: datapoints.cols,
        cols: 1,
        data: vec![T::from(1.0); datapoints.cols],
    };
    let mut x: DataFrame<T> = DataFrame {
        rows: datapoints.cols,
        cols: 1,
        data: vec![T::from(1.0); datapoints.cols],
    };

    while grad.len() > tol.into() {
        // calculate gradient

        let atax: DataFrame<T> = 2.0 * &matrix_multiplication(&matrix_multiplication(&datapoints.transpose(), &datapoints), &x);
        let aty: DataFrame<T> = 2.0 * &matrix_multiplication(&datapoints.transpose(), &target);

        grad = &atax - &aty;

        // update x
        x = &x - &(step_size * &grad);

    }

    x
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
