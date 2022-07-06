# pointMLP-new
尝试改进PointMLP, 进行的一些实验

实验如下：  
1. 随机采样  
原PointMLP每次取点云的前1024个点进行输入，该实验探究随机采样1024个点作为输入是否有影响  
2. Cosine loss
原PointMLP第一个stage通过FPS采样512个点，第二个stage通过FPS在512个点中采样256个点，接下来的stage以此递推。
这样忽略了另外512个未采样的点的信息，使用cosine loss，使另512个未采样的点对结果具有一定约束。
3. 随机采样 + Cosine loss
