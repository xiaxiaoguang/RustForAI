use super::Forward;
use super::Backward;

use rand::prelude::Distribution;
use rand::{Rng, distributions::uniform};
use rand::distributions::Uniform;
use ndarray::{self, Array2, Array1};
struct Linear{
    mat_w : Box<Array2<f32>>,
    mat_b : Box<Array1<f32>>,
    rc_x  : Box<Array1<f32>>,
}

impl Linear{
    fn new(inputsize : usize,outputsize : usize) -> Self{
        let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
        let uniform=Uniform::new(0.0,1.0);
        let w=Array2::from_shape_fn((inputsize,outputsize), |(_i,_j)| uniform.sample(&mut rng));
        let b = Array1::from_shape_fn(outputsize, |_i| uniform.sample(&mut rng));
        Self{
            mat_w : Box::new(w),
            mat_b : Box::new(b),
            rc_x  : Box::new(Array1::zeros(outputsize)),
        }
    }
}

impl Forward<Array1<f32>> for Linear{
    fn forward(&mut self, x:&Array1<f32>)-> Array1<f32> {
        self.rc_x = Box::new(x.clone());
        x.dot(&*self.mat_w)+ &*self.mat_b
    }
}

impl Backward