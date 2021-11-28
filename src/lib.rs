use std::collections::HashMap;
use std::f64::consts::SQRT_2;

use matrix::{ComplexDoubleMatrix, Matrix};

use matrix::BoxMatrix;
use num_complex::Complex64;
use rand::Rng;

pub mod matrix;
// max 32 bit(maximum of int)
struct Executer<T> {
    state: Box<dyn Matrix<Prec = T>>,
}

impl Executer<Complex64> {
    pub fn apply(&mut self, oper: QuantumOper<Box<dyn Matrix<Prec = Complex64>>>) {
        let func: Box<dyn Fn(BoxMatrix<Complex64>) -> BoxMatrix<Complex64>> = match oper.oper_type {
            GateType::I => {
                if oper.operand.len() != 1 {
                    panic!("Not correct Operand");
                }
                Box::new(move |x: BoxMatrix<Complex64>| x)
            }
            GateType::X => {
                if oper.operand.len() != 1 {
                    panic!("Not correct Operand");
                }
                let X = ComplexDoubleMatrix {
                    size: [2, 2],
                    data: vec![0.0.into(), 1.0.into(), 1.0.into(), 0.0.into()],
                };
                Box::new(move |x: BoxMatrix<Complex64>| X.mul(x))
            }
            GateType::Z => {
                if oper.operand.len() != 1 {
                    panic!("Not correct Operand");
                }
                let X = ComplexDoubleMatrix {
                    size: [2, 2],
                    data: vec![1.0.into(), 0.0.into(), 0.0.into(), (-1.0).into()],
                };
                Box::new(move |x: BoxMatrix<Complex64>| X.mul(x))
            }
            GateType::H => {
                if oper.operand.len() != 1 {
                    panic!("Not correct Operand");
                }
                let X = ComplexDoubleMatrix {
                    size: [2, 2],
                    data: vec![1.0.into(), 1.0.into(), 1.0.into(), (-1.0).into()],
                };
                let X = X.scalmul((1./SQRT_2).into());
                Box::new(move |x: BoxMatrix<Complex64>| X.mul(x))
            }
            GateType::CNot => {
                if oper.operand.len() != 2 {
                    panic!("Not correct Operand");
                }
                let X = ComplexDoubleMatrix {
                    size: [4, 4],
                    data: vec![
                                1.0.into(), 0.0.into(), 0.0.into(), 0.0.into(),
                                0.0.into(), 1.0.into(), 0.0.into(), 0.0.into(),
                                0.0.into(), 0.0.into(), 0.0.into(), 1.0.into(),
                                0.0.into(), 0.0.into(), 1.0.into(), 0.0.into()
                            ],
                };
                Box::new(move |x: BoxMatrix<Complex64>| X.mul(x))
            }
            _ => unimplemented!(),
        };
        let not_operand_mask: u32 = !oper
            .operand
            .iter()
            .map(|x: &u32| 1 << x)
            .fold(0, |acc, x| acc | x);
        let mut map: HashMap<u32, Vec<u32>> = HashMap::new();
        
        for i in 0..self.state.size()[0] {
            map.entry(i & not_operand_mask)
                .or_insert(Vec::new())
                .push(i);
        }
        let val = map.drain().into_iter().map(|(_, mut v)| {
            for i in oper.operand.iter().rev() {
                let mask = 1<<i;
                v.sort_by(|x, y| (x&mask).cmp(&(y&mask)));
            }

            let mut input: Vec<Complex64> = vec![0.0.into(); v.len()];
            for i in 0..v.len() {
                input[i]=self.state.data()[v[i] as usize];
            }
            let m = Box::new(ComplexDoubleMatrix {
                size: [v.len() as u32, 1],
                data: input
            });
            let a = func(m);
            let output = a.data();
            let ans: Vec<Complex64> = output.to_vec();
            let ret: Vec<(_, _)> = v.into_iter().zip(ans.into_iter()).collect();
            ret
        }).fold(vec![Complex64::from(0.0); self.state.size()[0] as usize], |mut acc, x| {
            for (i, v) in x {
                acc[i as usize] = v
            }
            acc
        });
        self.state = Box::new(
            ComplexDoubleMatrix {
                size: self.state.size(),
                data: val
            }
        )
    }

    pub fn measure(&mut self, bit: u32) -> bool{
        use rand::thread_rng;
        let mut rng = thread_rng();
        let mask: u32 = 1<<bit;
        let one : Vec<_> = self.state.data().into_iter().enumerate().filter(|(idx, x)| ((idx.clone() as u32) & mask)> 0).map(|(_,x)| x.clone()).collect();
        let one = ComplexDoubleMatrix {
            size: [self.state.size()[0]/2, 1],
            data: one
        };
        let zero: Vec<_> = self.state.data().into_iter().enumerate().filter(|(idx, x)| ((idx.clone() as u32) & mask)==0).map(|(_,x)| x.clone()).collect();
        let zero = ComplexDoubleMatrix {
            size: [self.state.size()[0]/2, 1],
            data: zero
        };
        let t: f64 = rng.gen_range(0.0..1.0);
        let prob_one = (one.norm()).powi(2);
        if t < prob_one {
            let one = one.scalmul((1./one.norm()).into());
            self.state = one;
            return true
        } else {
            let zero = zero.scalmul((1./zero.norm()).into());
            self.state = zero;
            return false
        }
    }
}

pub(crate) struct QuantumOper<T> {
    pub(crate) operand: Vec<u32>,
    pub(crate) oper_type: GateType<T>,
}

pub(crate) enum GateType<T> {
    I,
    X,
    Y,
    Z,
    H,
    CNot,
    U(T),
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use crate::{Executer, GateType, QuantumOper, matrix::ComplexDoubleMatrix};

    #[test]
    fn test_apply1() {
        let mut exc = Executer {
            state: Box::new(
                ComplexDoubleMatrix {
                    size: [4, 1],
                    data: vec![Complex64::from(0.0),
                    Complex64::from(0.0),
                    Complex64::from(1.0),
                    Complex64::from(0.0)]
                }
            )
        };
        exc.apply(QuantumOper{
            operand: vec![1,0],
            oper_type: GateType::CNot,
        });
        let ans: Vec<Complex64> = vec![
            0.0.into(),
            0.0.into(),
            0.0.into(),
            1.0.into(),
        ];
        assert_eq!(&ans, exc.state.data());
    }

    #[test]
    fn test_apply2() {
        let mut exc = Executer {
            state: Box::new(
                ComplexDoubleMatrix {
                    size: [16, 1],
                    data: vec![
                        Complex64::from(1.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0)
                    ]
                }
            )
        };
        exc.apply(QuantumOper{
            operand: vec![0],
            oper_type: GateType::H,
        });
        exc.apply(QuantumOper{
            operand: vec![1],
            oper_type: GateType::H,
        });
        exc.apply(QuantumOper{
            operand: vec![2],
            oper_type: GateType::H,
        });
        exc.apply(QuantumOper{
            operand: vec![3],
            oper_type: GateType::H,
        });
        // println!("{:?}", &exc.state.data());
        exc.apply(QuantumOper{
            operand: vec![0],
            oper_type: GateType::H,
        });
        exc.apply(QuantumOper{
            operand: vec![1],
            oper_type: GateType::H,
        });
        exc.apply(QuantumOper{
            operand: vec![2],
            oper_type: GateType::H,
        });
        exc.apply(QuantumOper{
            operand: vec![3],
            oper_type: GateType::H,
        });
        // println!("{:?}", &exc.state.data());
        let data = exc.state.data();
        assert!(data[0].norm() > 0.99.into());
        for i in 1..16 {
            assert!(data[i].norm() < 0.01.into());
        }
    }

    #[test]
    fn test_apply3() {
        let mut exc = Executer {
            state: Box::new(
                ComplexDoubleMatrix {
                    size: [4, 1],
                    data: vec![
                        Complex64::from(1.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                    ]
                }
            )
        };
        exc.apply(QuantumOper{
            operand: vec![0],
            oper_type: GateType::H,
        });
        exc.apply(QuantumOper{
            operand: vec![0,1],
            oper_type: GateType::CNot,
        });
        assert_eq!(exc.state.data()[1], 0.0.into());
        assert_eq!(exc.state.data()[2], 0.0.into());
    }

    #[test]
    fn test_measure() {
        let mut exc = Executer {
            state: Box::new(
                ComplexDoubleMatrix {
                    size: [4, 1],
                    data: vec![
                        Complex64::from(1.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                        Complex64::from(0.0),
                    ]
                }
            )
        };
        exc.apply(QuantumOper{
            operand: vec![0],
            oper_type: GateType::H,
        });
        exc.apply(QuantumOper{
            operand: vec![0,1],
            oper_type: GateType::CNot,
        });
        let m1 = &exc.measure(0);
        let m2 = &exc.measure(0);
        assert!(!(m1^m2));
    }
}