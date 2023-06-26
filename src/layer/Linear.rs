use super::Forward;
use rand::Rng;

struct Linear{
    mat_w : Box<Vec<Vec<f32>>>,
    mat_b : Box<Vec<f32>>,
}
impl Linear{
    fn new(inputsize : usize,outputsize : usize) -> Self{
        let mut rng = rand::thread_rng();

        Self{
            Box::new(vec![vec![rng.gen::<f32>();inputsize];outputsize]),
            Box::new(vec![rng.gen::<f32>()]),
        }
    }
}
