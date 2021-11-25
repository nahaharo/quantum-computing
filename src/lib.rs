use std::convert::TryInto;

use cblas::Layout::ColumnMajor;
use cblas::Transpose;
use num_complex::Complex64;

pub trait Matrix<Prec> {
    fn presicion(&self) -> Presicion;
    fn size(&self) -> [u32; 2];
    fn data(&self) -> &Vec<Prec>;
    fn multiplicate<M: Matrix<Prec>>(&self, other: M) -> Self;
    fn tensor_product<M: Matrix<Prec>>(&self, other: M) -> Self;
    fn at(&self, i: u32, j: u32) -> Prec;
    fn row(&self, i: u32) -> Vec<Prec>;
    fn col(&self, j: u32) -> Vec<Prec>;
}

pub enum Presicion {
    Single,
    Double,
    SingleComplex,
    DoubleComplex,
}

pub struct ComplexDoubleMatrix {
    pub size: [u32; 2],
    pub data: Vec<Complex64>,
}

impl Matrix<Complex64> for ComplexDoubleMatrix {
    fn presicion(&self) -> Presicion {
        Presicion::DoubleComplex
    }
    fn size(&self) -> [u32; 2] {
        self.size
    }
    fn multiplicate<M: Matrix<Complex64>>(&self, other: M) -> Self {
        use cblas::zgemm;
        let [m, k] = other.size();
        let [k1, n] = self.size;
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
                &**other.data(),
                m as i32,
                &*self.data,
                k as i32,
                Complex64::from(1.0),
                &mut ans,
                m as i32,
            );
        }
        return ComplexDoubleMatrix {
            size: [m, k],
            data: ans,
        };
    }

    fn data(&self) -> &Vec<Complex64> {
        &self.data
    }

    fn tensor_product<M: Matrix<Complex64>>(&self, other: M) -> Self {
        todo!()
    }

    fn at(&self, i: u32, j: u32) -> Complex64 {
        if i >= self.size[0] || j >= self.size[1] {
            panic!("Overflow");
        }
        self.data[(j*self.size[0]+i) as usize]
    }

    fn row(&self, i: u32) -> Vec<Complex64> {
        todo!()
    }

    fn col(&self, j: u32) -> Vec<Complex64> {
        self.data()[((j*self.size[0]) as usize)..(((j+1)*self.size[0]) as usize)].to_vec()
    }
}

pub(crate) struct QuatOper<T> {
    pub(crate) operand: Vec<u32>,
    pub(crate) oper_type: GateType<T>,
}

pub(crate) enum GateType<T> {
    X,
    Y,
    Z,
    H,
    CNot,
    U(T),
}

#[cfg(test)]
mod tests {
    use num_complex::{Complex64};

    use crate::{ComplexDoubleMatrix, Matrix};

    #[test]
    fn test_mul() {
        let A = ComplexDoubleMatrix {
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
        let B = ComplexDoubleMatrix {
            size: [3, 4],
            data: vec![
                1.0.into(),
                5.0.into(),
                9.0.into(),
                2.0.into(),
                6.0.into(),
                10.0.into(),
                3.0.into(),
                7.0.into(),
                11.0.into(),
                4.0.into(),
                8.0.into(),
                12.0.into(),
            ],
        };
        let C = B.multiplicate(A);
        let ans: Vec<Complex64> = vec![
            38.0.into(),
            83.0.into(),
            44.0.into(),
            98.0.into(),
            50.0.into(),
            113.0.into(),
            56.0.into(),
            128.0.into(),
        ];
        assert_eq!(&ans, C.data());
    }
    #[test]
    fn test_at() {
        let A = ComplexDoubleMatrix {
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
        assert_eq!(A.at(0,0), Complex64::from(1.0));
        assert_eq!(A.at(1,0), Complex64::from(4.0));
        assert_eq!(A.at(0,1), Complex64::from(2.0));
    }
    #[test]
    fn test_col() {
        let A = ComplexDoubleMatrix {
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
        let ans: Vec<Complex64> = vec![
            2.0.into(),
            5.0.into(),
        ];
        assert_eq!(A.col(1), ans);
    }
}
