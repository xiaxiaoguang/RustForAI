use super::Forward;
use super::Backward;

use ndarray::ArrayView1;
use rand::prelude::Distribution;
use rand::distributions::Uniform;
use ndarray::{self, Array2, Axis};

pub struct Linear2D{
    mat_w : Box<Array2<f32>>,
    mat_b : Box<Array2<f32>>,
    rc_x  : Box<Array2<f32>>,
    rc_w  : Box<Array2<f32>>,
    rc_b  : Box<Array2<f32>>,
}

impl Linear2D{
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

fn batch_mul(x : &Array2<f32>,f : fn(w : ArrayView1<f32>)->Array2<f32>,outputsize : usize) -> Array2<f32> {
    let source = x.map_axis(Axis(1), |x| f(x).into_shape(outputsize).unwrap());
    let mut target = Array2::<f32>::zeros((x.shape()[0],outputsize));
    for (i,value) in source.iter().enumerate() {
        for (j , a) in value.iter().enumerate() {
            target[[i,j]] = *a;
        }
    }
    target
}

// 大小永远都是两维的，因为我们要实现转置操作
impl Forward<Array2<f32>> for Linear2D{
    fn forward(&mut self, x:&Array2<f32>)-> Array2<f32> {
        self.rc_x = Box::new(x.clone());
        // batch_mul(x, |x|(self.mat_w.dot(&x)+&*self.mat_b), self.mat_w.shape()[0])
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

impl Backward<Array2<f32>> for Linear2D{
    fn backward(&mut self,x : &Array2<f32>)->Array2<f32>{
        let x_1 = x.map_axis(Axis(1), |x| x);
        let x_2= self.rc_x.map_axis(Axis(1), |x| x);
        let mut flg: bool=true;
        let mut tmp = Array2::<f32>::zeros((1,1)); 
        // 这里形状并不真实，因为我只是为了凑齐这个初始化！
        for (&matx,&matb) in x_1.iter().zip(x_2.iter()){
            if flg {
                tmp = matx.into_shape((matx.len(),1)).unwrap().dot(&(matb.into_shape((1,matb.len())).unwrap()));
                flg = false;
            }
            else {
                tmp = tmp + matx.into_shape((matx.len(),1)).unwrap().dot(&(matb.into_shape((1,matb.len())).unwrap()));
            }

        }
        self.rc_w = Box::new(tmp / (x.shape()[0] as f32));
        let mut tmp = Array2::<f32>::zeros((1,1));
        flg = true;
        for &x in x_1.iter(){
            if flg {
                tmp = x.into_shape((x.len(),1)).unwrap().to_owned();
                flg=false;
            }else {
                tmp = tmp + x;
            }
        }
        self.rc_b = Box::new(tmp / (x.shape()[0] as f32));
        let source = x.map_axis(Axis(1),|x|self.mat_w.t().dot(&x));
        let mut target = Array2::<f32>::zeros((x.shape()[0],self.mat_w.shape()[0]));
        for (i,values) in source.iter().enumerate(){
            for (j,a) in values.iter().enumerate(){
                target[[i,j]] = *a;
            }

        }
        target
    }
    fn sgd(&mut self,lr : f32,mm : f32) {
        self.mat_w = Box::new(*self.mat_w * mm + *self.rc_w * lr * (1.0-mm));
        self.mat_b = Box::new(*self.mat_b * mm + *self.rc_b * lr * (1.0-mm));
    }
    fn outrc(&self) -> Array2<f32> {
        (*self.rc_x).clone()
    }
}