use std::marker::PhantomData;

use ndarray::Array2;

use super::module::Model;
use super::layer::Backward;
use super::layer::Forward;

struct Optim
{
    model : Box<Model>,
    lr    : f32,
    moment: f32,
}

impl Optim
{
    fn new(m : Model,l : f32,p : f32) -> Self{
        Self{
            model : Box::new(m),
            lr    : l,
            moment: p,
        }
    }
    fn one_step(&mut self,a : Array2<f32>){
        self.model.sgd(a,self.lr,self.moment)
    }
}