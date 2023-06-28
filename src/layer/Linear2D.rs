use super::Forward;
use super::Backward;

use rand::prelude::Distribution;
use rand::distributions::Uniform;
use ndarray::{self, Array2, Axis};

struct Linear{
    mat_w : Box<Array2<f32>>,
    mat_b : Box<Array2<f32>>,
    rc_x  : Box<Array2<f32>>,
    rc_w  : Box<Array2<f32>>,
    rc_b  : Box<Array2<f32>>,
}

impl Linear{
    fn new(inputsize : usize,outputsize : usize) -> Self{
        let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
        let uniform=Uniform::new(0.0,1.0);
        Self{
            mat_w : Box::new(Array2::from_shape_fn((outputsize,inputsize), |(_i,_j)| uniform.sample(&mut rng))),
            mat_b : Box::new(Array2::from_shape_fn((outputsize,1), |_i| uniform.sample(&mut rng))),
            rc_x  : Box::new(Array2::zeros((1,inputsize))),
            rc_w  : Box::new(Array2::zeros((inputsize,outputsize))),
            rc_b  : Box::new(Array2::zeros((outputsize,1))),
        }
    }
}


/// 大小永远都是两维的，因为我们要实现转置操作
impl Forward<Array2<f32>> for Linear{
    fn forward(&mut self, x:&Array2<f32>)-> Array2<f32> {
        self.rc_x = Box::new(x.clone());
        // self.mat_w.dot(x) + &*self.mat_b
        let source = x.map_axis(Axis(1),
            |x| 
                (self.mat_w.dot(&x)+ &*self.mat_b).into_shape(self.mat_w.shape()[0]).unwrap()
            );
        let mut target = Array2::<f32>::zeros((x.shape()[0],self.mat_w.shape()[0]));
        for (i,value) in source.iter().enumerate(){
            for (j,a) in value.iter().enumerate(){
                target[[i,j]]=*a;
            }
        }
        target
    }
}

impl Backward<Array2<f32>> for Linear{
    fn backward(&mut self,x : &Array2<f32>)->Array2<f32>{
        
    }
}