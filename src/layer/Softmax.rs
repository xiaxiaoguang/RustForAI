use super::Forward;
use super::Backward;

use rand::prelude::Distribution;
use rand::distributions::Uniform;
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

}

impl Backward<Array2<f32>> for Softmax{
    
}