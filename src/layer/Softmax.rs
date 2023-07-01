use super::Forward;
use super::Backward;

use ndarray::{self, Array2};

pub struct Softmax{
    rc_x : Box<Array2<f32>>,
    ot_x : Box<Array2<f32>>,
}

impl Softmax{
    fn new()->Self{
        Self{
            rc_x : Box::new(Array2::zeros((1,1))),
            ot_x : Box::new(Array2::zeros((1,1))),
        }
    }
}

impl Forward<Array2<f32>> for Softmax{
    fn forward(&mut self, x:&Array2<f32>)-> Array2<f32> {
        let tmp = x.mapv(f32::exp);
        let mut s : f32 = 0.0;
        for i in tmp.iter(){
            s += i;
        }
        tmp/s
    }
}
impl Backward<Array2<f32>> for Softmax{
    fn backward(&mut self, x:&Array2<f32>)-> Array2<f32> {
        (*self.ot_x).clone()-x
    }   
    fn sgd(&mut self,_lr : f32,_mm : f32) {
        return ;
    }
    fn outrc(&self) -> Array2<f32> {
        (*self.ot_x).clone()
    }
}