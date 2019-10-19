#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn test_array_addition() {
        use crate::array_addition;

        assert_eq!(array_addition(&[1, 2], &[2, 1]), &[3, 3])
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