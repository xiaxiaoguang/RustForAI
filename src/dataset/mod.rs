use ndarray::{Array1, arr1};

struct Dataset<T,U>{
    data : Array1<T>,
    label : Array1<U>,
    batchsize : usize,
    nuse : usize,
}

impl<T,U> Dataset<T,U> 
where T : Clone,U : Clone
{
    fn new(d : Array1<T>,l : Array1<U>,b : usize)->Self {
        assert!(d.len()!=l.len());
        Self{
            data : d,
            label : l,
            batchsize : b,
            nuse : 0,
        }
    }
    fn len(&self)->usize{
        self.data.len()
    }
    fn next(&mut self)->(Array1<T>,Array1<U>){
        let mut i = self.nuse;
        
        let mut fordata = vec![];
        let mut forlabel = vec![];
        while i <self.nuse + self.batchsize {
            fordata.push(self.data[i].clone());
            forlabel.push(self.label[i].clone());
            i += 1;
        }
        let a = arr1(fordata.as_slice());
        let b = arr1(forlabel.as_slice());
        self.nuse = i;
        (a,b)
    }
}
