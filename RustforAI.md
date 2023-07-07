# Rust_for_AI

利用rust语言实现一个最基础的AI系统,春季学期rust语言实践项目

Producer : xiaxiaoguang(loveJY)

[github项目地址](https://github.com/xiaxiaoguang/RustForAI)

## 项目介绍

目的是仿生pytorch那样实现一个mnist手写数字图像识别任务。

实现了Layer/Loss/Module/Optimizer/Dataset几个mod分支，分别与pytorch中原有的属性相对应

### Layer

实现了Linear/ReLu/Linear2D，代表线性层和relu激活函数层

Linear2D表示支持设置非1的batchsize的训练

### Loss

loss function为softmax激活+cross_entrophy_loss结合，这样能有效减少写梯度反向传递时的代码量。

### module

封装layer，实现layer之间的forward和backwards

### optimize

简单的SGD实现

### dataset

手动处理mnist数据集，完成数据分析的任务并构建训练集/测试集

### main

实现完整的训练过程*beta*，此时还没有完成这部分代码。

## mainCrateAboutThisProject

[rand](https://docs.rs/rand/latest/rand/)

用于生成随机数,使用样例：

```rust
use rand::prelude::*;

let mut rng = rand::thread_rng();
let y: f64 = rng.gen(); // generates a float between 0 and 1

let mut nums: Vec<i32> = (1..100).collect();
nums.shuffle(&mut rng);
```

全部随机过程在于ThreadRng类型进行生成。

**distribution**可以返回各种类型分布的随机数

uniform是正态分布，往往用来初始化网络参数信息，可以如下进行：

```rust

use rand::distributions::{Distribution, Uniform};

let between = Uniform::<f32>::new(0.0,1.0);
let mut rng = rand::thread_rng();
let mut sum = 0;
for _ in 0..1000 {
    sum += between.sample(&mut rng);
}
println!("{}", sum);

```

[ndarray](https://docs.rs/ndarray/0.15.6/ndarray/)

用于高维数组，类似python中的numpy

好处是方便创建与矩阵运算（转置，点积等），实现了封装好的函数。

## relativeCrate

rustlearn

包含经典的机器学习算法，决策树那些，可供比较

tangram

与上述相似，提供易于使用的机器学习框架

leaf

构建/评估机器学习模型的框架 决策树，随机森林神经网络等等

tract

深度学习推理库，支持各种网络架构和硬件平台，这里只有运载训练好的模型而没有训练过程

rulinarg

线性代数的计算库

