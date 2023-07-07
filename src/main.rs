mod layer;
mod dataset;
mod module;
mod optimizer;
mod Loss;

use rand::distributions::{Distribution, Uniform};

fn main() {
    let between = Uniform::<f32>::new(0.0,1.0);
    let mut rng = rand::thread_rng();
    let mut sum = 0.0;
    for _ in 0..1000 {
        sum += between.sample(&mut rng);
    }
    println!("{}", sum);
}
