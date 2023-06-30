use ndarray::Array2;
use super::layer::Forward;
use super::layer::Backward;
use super::layer::Layer;


pub struct Model
{
    list : Box<Vec<Layer>>,
}

impl Model
{
    fn new()->Self{
        Self{
            list : Box::new(vec![]),
        }
    }
    fn add(&mut self,a :Layer){
        self.list.push(a)
    }
    fn calculate(&mut self,mut a : Array2<f32>) -> Array2<f32> {
        for i in self.list.iter_mut(){
            a = i.forward(&a);
        }
        a
    }
    fn sgd(&mut self,a :U,lr : f32,mm : f32){

        for i in self.list.iter_mut(){
            
        }
    }
}