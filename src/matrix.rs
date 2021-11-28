use std::convert::TryInto;

use cblas::Layout::ColumnMajor;
use cblas::Transpose;
use num_complex::Complex64;

pub type BoxMatrix<T> = Box<dyn Matrix<Prec = T>>;

pub trait Matrix {
    type Prec;

    fn presicion(&self) -> Presicion;
    fn size(&self) -> [u32; 2];
    fn data(&self) -> &Vec<Self::Prec>;
    fn mul(&self, other: BoxMatrix<Self::Prec>) -> BoxMatrix<Self::Prec>;
    fn scalmul(&self, scalar: Complex64) -> BoxMatrix<Self::Prec>;
    fn kron(&self, other: BoxMatrix<Self::Prec>) -> BoxMatrix<Self::Prec>;
    fn norm(&self) -> f64;
    fn at(&self, i: u32, j: u32) -> Self::Prec;
    fn row(&self, i: u32) -> Vec<Self::Prec>;
    fn col(&self, j: u32) -> Vec<Self::Prec>;
}

pub enum Presicion {
    Single,
    Double,
    SingleComplex,
    DoubleComplex,
}

#[derive(Debug)]
pub struct ComplexDoubleMatrix {
    pub size: [u32; 2],
    pub data: Vec<Complex64>,
}

impl Matrix for ComplexDoubleMatrix {
    type Prec = Complex64;

    fn presicion(&self) -> Presicion {
        Presicion::DoubleComplex
    }
    fn size(&self) -> [u32; 2] {
        self.size
    }

    fn scalmul(&self, scalar: Complex64) -> BoxMatrix<Self::Prec> {
        use cblas::zscal;
        let mut ans = self.data.clone();
        unsafe {
            zscal((self.size[0] * self.size[1]) as i32, scalar, &mut *ans, 1);
        }
        Box::new(ComplexDoubleMatrix {
            size: self.size,
            data: ans,
        })
    }

    fn mul(&self, other: BoxMatrix<Self::Prec>) -> BoxMatrix<Self::Prec> {
        use cblas::zgemm;
        let [m, k] = self.size;
        let [k1, n] = other.size();
        if k != k1 {
            panic!("Not correct matrix dimension");
        }
        let mut ans: Vec<Complex64> = vec![Complex64::from(0.); (m * n).try_into().unwrap()];

        unsafe {
            zgemm(
                ColumnMajor,
                Transpose::None,
                Transpose::None,
                m as i32,
                n as i32,
                k as i32,
                Complex64::from(1.0),
                &*self.data,
                m as i32,
                &**other.data(),
                k as i32,
                Complex64::from(1.0),
                &mut ans,
                m as i32,
            );
        }
        Box::new(ComplexDoubleMatrix {
            size: [m, n],
            data: ans,
        })
    }

    fn data(&self) -> &Vec<Complex64> {
        &self.data
    }

    fn kron(&self, other: BoxMatrix<Self::Prec>) -> BoxMatrix<Self::Prec> {
        let mut ans: Vec<Complex64> =
            vec![Complex64::from(0.); self.data.len() * other.data().len()];
        let [n1, m1] = self.size;
        let [n2, m2] = other.size();
        for i1 in 0..n1 {
            for j1 in 0..m1 {
                for i2 in 0..n2 {
                    for j2 in 0..m2 {
                        let i = i1 * n2 + i2;
                        let j = j1 * m2 + j2;
                        ans[(i + (n1 * n2) * j) as usize] = self.data[(i1 + n1 * j1) as usize]
                            * other.data()[(i2 + n2 * j2) as usize];
                    }
                }
            }
        }

        Box::new(ComplexDoubleMatrix {
            size: [
                self.size[0] * other.size()[0],
                self.size[1] * other.size()[1],
            ],
            data: ans,
        })
    }

    fn norm(&self) -> f64 {
        use cblas::zdotc_sub;
        let mut val: Vec<Complex64> = vec![0.0.into(); 1];
        unsafe {
            zdotc_sub(self.data.len() as i32, &*self.data, 1, &*self.data, 1, &mut *val);
        }
        val[0].norm().sqrt()
    }

    fn at(&self, i: u32, j: u32) -> Complex64 {
        if i >= self.size[0] || j >= self.size[1] {
            panic!("Overflow");
        }
        self.data[(j * self.size[0] + i) as usize]
    }

    fn row(&self, i: u32) -> Vec<Complex64> {
        if i >= self.size[0] {
            panic!("Overflow");
        }
        let mut ans: Vec<Complex64> = vec![Complex64::from(0.); self.size[1].try_into().unwrap()];

        for ans_idx in 0..self.size[1] {
            ans[ans_idx as usize] = self.data[(ans_idx * self.size[0] + i) as usize];
        }
        ans
    }

    fn col(&self, j: u32) -> Vec<Complex64> {
        if j >= self.size[1] {
            panic!("Overflow");
        }
        self.data()[((j * self.size[0]) as usize)..(((j + 1) * self.size[0]) as usize)].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use crate::matrix::{ComplexDoubleMatrix, Matrix};

    #[test]
    fn test_mul() {
        let a = ComplexDoubleMatrix {
            size: [4,3],
            data: vec![
                1.0.into(),
                4.0.into(),
                7.0.into(),
                10.0.into(),
                2.0.into(),
                5.0.into(),
                8.0.into(),
                11.0.into(),
                3.0.into(),
                6.0.into(),
                9.0.into(),
                12.0.into(),
            ],
        };
        let b = ComplexDoubleMatrix {
            size: [3, 2],
            data: vec![
                1.0.into(),
                3.0.into(),
                5.0.into(),
                2.0.into(),
                4.0.into(),
                6.0.into(),
            ],
        };

        let c = a.mul(Box::new(b));
        let ans: Vec<Complex64> = vec![
            22.0.into(),
            49.0.into(),
            76.0.into(),
            103.0.into(),
            28.0.into(),
            64.0.into(),
            100.0.into(),
            136.0.into(),
        ];
        assert_eq!(&ans, c.data());
    }

    #[test]
    fn test_scalmul() {
        let a = ComplexDoubleMatrix {
            size: [3, 3],
            data: vec![
                1.0.into(),
                4.0.into(),
                7.0.into(),
                2.0.into(),
                5.0.into(),
                8.0.into(),
                3.0.into(),
                6.0.into(),
                9.0.into(),
            ],
        };

        let b = a.scalmul(Complex64::from(4.));
        let ans: Vec<Complex64> = vec![
            4.0.into(),
            16.0.into(),
            28.0.into(),
            8.0.into(),
            20.0.into(),
            32.0.into(),
            12.0.into(),
            24.0.into(),
            36.0.into(),
        ];
        assert_eq!(&ans, b.data());
    }

    #[test]
    fn test_tp() {
        let a = ComplexDoubleMatrix {
            size: [2, 2],
            data: vec![1.0.into(), 3.0.into(), 2.0.into(), 4.0.into()],
        };
        let b = ComplexDoubleMatrix {
            size: [2, 2],
            data: vec![5.0.into(), 7.0.into(), 6.0.into(), 8.0.into()],
        };

        let c = a.kron(Box::new(b));
        let ans: Vec<Complex64> = vec![
            5.0.into(),
            7.0.into(),
            15.0.into(),
            21.0.into(),
            6.0.into(),
            8.0.into(),
            18.0.into(),
            24.0.into(),
            10.0.into(),
            14.0.into(),
            20.0.into(),
            28.0.into(),
            12.0.into(),
            16.0.into(),
            24.0.into(),
            32.0.into(),
        ];
        assert_eq!(&ans, c.data());
    }

    #[test]
    fn test_norm() {
        let a = ComplexDoubleMatrix {
            size: [2,1],
            data: vec![
                3.0.into(),
                4.0.into(),
            ],
        };
        assert_eq!(a.norm(), 5.);
    }

    #[test]
    fn test_at() {
        let a = ComplexDoubleMatrix {
            size: [2, 3],
            data: vec![
                1.0.into(),
                4.0.into(),
                2.0.into(),
                5.0.into(),
                3.0.into(),
                6.0.into(),
            ],
        };
        assert_eq!(a.at(0, 0), Complex64::from(1.0));
        assert_eq!(a.at(1, 0), Complex64::from(4.0));
        assert_eq!(a.at(0, 1), Complex64::from(2.0));
    }
    #[test]
    fn test_col() {
        let a = ComplexDoubleMatrix {
            size: [2, 3],
            data: vec![
                1.0.into(),
                4.0.into(),
                2.0.into(),
                5.0.into(),
                3.0.into(),
                6.0.into(),
            ],
        };
        let ans: Vec<Complex64> = vec![2.0.into(), 5.0.into()];
        assert_eq!(a.col(1), ans);
    }
    #[test]
    fn test_row() {
        let a = ComplexDoubleMatrix {
            size: [2, 3],
            data: vec![
                1.0.into(),
                4.0.into(),
                2.0.into(),
                5.0.into(),
                3.0.into(),
                6.0.into(),
            ],
        };
        let ans: Vec<Complex64> = vec![1.0.into(), 2.0.into(), 3.0.into()];
        assert_eq!(a.row(0), ans);
    }
}
