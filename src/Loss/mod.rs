use ndarray::{self, Array2};


pub struct CrossEntrophyLoss{
    ot_x : Box<Array2<f32>>,
}