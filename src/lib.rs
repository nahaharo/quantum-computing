use std::convert::TryInto;

use num_complex::Complex64;
use blas::zgemm;

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
    size: [u32; 2],
    data: Vec<Complex64>,
}

impl Matrix<Complex64> for ComplexDoubleMatrix {
    fn presicion(&self) -> Presicion {
        Presicion::DoubleComplex
    }
    fn size(&self) -> [u32; 2] {
        self.size
    }
    fn multiplicate<M: Matrix<Complex64>>(&self, other: M) -> Self {
        use blas::zgemm;
        let [m, k] = other.size();
        let [k1, n] = self.size;
        if k != k1 {
            panic!("Not correct matrix dimension");
        }
        let mut ans: Vec<Complex64> = vec![Complex64::from(0.); (m * k).try_into().unwrap()];

        unsafe {
            zgemm(
                b'N',
                b'N',
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
        todo!()
    }

    fn tensor_product<M: Matrix<Complex64>>(&self, other: M) -> Self {
        todo!()
    }

    fn at(&self, i: u32, j: u32) -> Complex64 {
        todo!()
    }

    fn row(&self, i: u32) -> Vec<Complex64> {
        todo!()
    }

    fn col(&self, j: u32) -> Vec<Complex64> {
        todo!()
    }
}

pub struct QuatOper<T> {
    operType: GateType<T>,
}

pub enum GateType<T> {
    X,
    Y,
    Z,
    H,
    CNot,
    U(T),
}

#[cfg(test)]
mod tests {
    use num_complex::{Complex, Complex64};

    use crate::{ComplexDoubleMatrix, Matrix};

    #[test]
    fn multiplicate() {
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
            ]};
        let C = A.multiplicate(B);
        assert_eq!(Complex64::from(40.0), C.data()[0]);
    }
}
