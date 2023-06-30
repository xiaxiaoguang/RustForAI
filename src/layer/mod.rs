use ndarray::Array2;

mod Linear;
mod ReLU;
mod Linear2D;

pub enum Layer {
    Linear(Linear::Linear),
    ReLU(ReLU::ReLU),
    Linear2D(Linear2D::Linear2D),
}

pub trait Forward<T>{
    fn forward(&mut self,x:&T)-> T;
}

pub trait Backward<T>{
    fn backward(&mut self,x:&T) -> T;
}

impl Forward<Array2<f32>> for Layer{
    fn forward(&mut self,x:&Array2<f32>)-> Array2<f32> {
        self.forward(x)
    }
}

impl Backward<Array2<f32>> for Layer{
    fn backward(&mut self,x:&Array2<f32>) -> Array2<f32> {
        self.backward(x)   
    }
}