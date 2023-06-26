use ndarray::Array;

// Usage example of move_into in safe code
#[cfg(test)]

fn main(){
    let mut a = Array::default((10, 10));
    println!("{:?}",a);
    let b = Array::from_shape_fn((10, 10), |(i, j)| (i + j).to_string());
    b.move_into(&mut a);
}
