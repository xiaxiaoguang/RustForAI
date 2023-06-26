mod Linear;

trait Forward<T>{
    fn forward(&mut self,x:&T)-> T;
}

trait Backward<T>{
    fn backward(&mut self,x:&T);

}