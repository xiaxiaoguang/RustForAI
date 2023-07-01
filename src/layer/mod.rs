use ndarray::Array2;

mod Linear;
mod ReLU;
mod Linear2D;
mod Softmax;

pub enum Layer {
    Linear(Linear::Linear),
    ReLU(ReLU::ReLU),
    Linear2D(Linear2D::Linear2D),
    Softmax(Softmax::Softmax),
}

pub trait Forward<T>{
    fn forward(&mut self,x:&T)-> T;
}

pub trait Backward<T>{
    fn backward(&mut self,x:&T) -> T;
    fn sgd(&mut self,lr : f32,mm : f32);
    fn outrc(&self) -> T;
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
    fn sgd(&mut self,lr : f32,mm : f32) {
        self.sgd(lr, mm)
    }
    fn outrc(&self) -> Array2<f32> {
        self.outrc()
    }
}