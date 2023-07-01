use super::Forward;
use super::Backward;
use ndarray::{self, Array2};

pub struct ReLU{
    rc_x : Box<Array2<f32>>,
}

impl ReLU{
    fn new(inputsize:usize)->Self{
        Self{
            rc_x : Box::new(Array2::zeros((inputsize,1))),
        }
    }
}

impl Forward<Array2<f32>> for ReLU{
    fn forward(&mut self,x : &Array2<f32>)->Array2<f32>{
        self.rc_x = Box::new(x.clone());
        Array2::from_shape_fn((x.shape()[0],x.shape()[1]),
        |(i,j)| 
            if x[[i,j]] > 0.0{
                x[[i,j]]
            }else {
                0.0
            }
        )
    }
}

impl Backward<Array2<f32>> for ReLU {
    fn backward(&mut self,x:&Array2<f32>) -> Array2<f32> {
        Array2::from_shape_fn((x.shape()[0],x.shape()[1]),
        |(i,j)| 
            if self.rc_x[[i,j]]>0.0{
                x[[i,j]]
            }else {
                0.0
            }
        )
    }
    fn sgd(&mut self,_lr : f32,_mm : f32) {
        return ;
    }
    fn outrc(&self) -> Array2<f32> {
        (*self.rc_x).clone()
    }
}