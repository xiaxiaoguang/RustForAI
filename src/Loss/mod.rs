use ndarray::{self, Array2};


pub struct CrossEntrophyLoss{
    ot_x : Box<Array2<f32>>,
}

fn gmax(a : f32, b : f32)->f32{
    if a>b {return a;}
    return b;
}

fn max(x : &Array2<f32>)->f32{
    let mut a = 0.0;
    for i in x.iter(){
        a = gmax(a,*i);
    }
    a
}

impl CrossEntrophyLoss{
    fn new(qwq : &Array2<f32>)->CrossEntrophyLoss{
        let mut x = qwq.clone();
        x -= max(&x);
        x  = x.map(|x|x.exp());
        x /= x.sum();
        Self{
            ot_x : Box::new(x),
        }
    }
    fn backward(&self,st_x : &Array2<f32>)->Array2<f32>{
        (&(*self.ot_x) - st_x).clone()
    }
}

