mod Linear;

trait Forward<T>{
    fn forward(&self,x: T)-> T;
}
