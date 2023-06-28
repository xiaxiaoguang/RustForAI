use ndarray::{Array3,Array2, Axis,ArrayBase,OwnedRepr,Dim};
// #[cfg(test)]

pub fn main() {
    let mut array = Array2::<i32>::zeros((3, 2));

    // Fill the array with consecutive numbers for demonstration
    for (i, elem) in array.iter_mut().enumerate() {
        *elem = i as i32;
    }

    println!("Original Array:\n{:?}", array);
    let multi =    Array2::<i32>::ones((2, 4));
    // Using map on the first dimension to multiply each element by 2


    let source = array.map_axis(Axis(1), |x| 
        // println!("{:?} {:?}",x,x.shape());
        (x.dot(&multi)));
    let mut target =ArrayBase::<OwnedRepr<i32>, Dim<[usize; 2]>>::zeros((3,4));

    for (i,a) in source.iter().enumerate(){
        for (j,values) in a.iter().enumerate(){
            target[[i,j]]=*values;
        }
    }
    println!("Mapped Array:\n{:?}", target);
}
