// use std::ops::Index;
use std::ops::Mul;
use std::ops::Add;
use std::iter::Sum;
use std::ops::Sub;
use std::ops::Div;
use std::ops::AddAssign;
// use std::cmp::PartialEq;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

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
            data: [1, 2, 0, 1].to_vec(),
        };

        let b = DataFrame {
            rows: 2,
            cols: 2,
            data: [2, 1, 1, 0].to_vec(),
        };

        let ab = DataFrame {
            rows: 2,
            cols: 2,
            data: [4, 1, 1, 0].to_vec(),
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
            data: [1, 2, 1, 0, 1, 1].to_vec(),
        };

        let b = DataFrame {
            rows: 3,
            cols: 2,
            data: [2, 1, 1, 0, 3, 2].to_vec(),
        };

        let ab = DataFrame {
            rows: 2,
            cols: 2,
            data: [7, 3, 4, 2].to_vec(),
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
            data: [1, 2, 0, 1].to_vec(),
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
            data: [1, 2, 0, 1].to_vec(),
        };

        let a_transposed = DataFrame {
            rows: 2,
            cols: 2,
            data: [1, 0, 2, 1].to_vec(),
        };

        assert_eq!(a.transpose(), a_transposed);
    }

    #[test]
    fn test_matrix_transposition_rect() {
        use crate::DataFrame;

        let a = DataFrame {
            rows: 2,
            cols: 3,
            data: [1, 2, 0, 1, 3, 2].to_vec(),
        };

        let a_transposed = DataFrame {
            rows: 3,
            cols: 2,
            data: [1, 1, 2, 3, 0, 2].to_vec(),
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
            data: [3, 1, 4, 2].to_vec(),
        };

        let lu = DataFrame {
            rows: 2,
            cols: 2,
            data: [3, 1, 4 / 3, 2 / 3].to_vec(),
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
            data: [1, 2, 3, 4, 5, 6, 7, 8, 9].to_vec(),
        };

        let lu = DataFrame {
            rows: 3,
            cols: 3,
            data: [1, 2, 3, 4, -3, -6, 7, 2, 0].to_vec(),
        };

        assert_eq!(lu_decompose(&b), lu);
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
