use rand::distributions::{Distribution, Uniform};

fn main(){
    let between = Uniform::from(0..1);
    let mut rng = rand::thread_rng();
    let mut sum = 0;
    for _ in 0..1000 {
        sum += between.sample(&mut rng);
    }
    println!("{}", sum);
}