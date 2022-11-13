<p align="center">PSOCLASSES</p>

介绍由中文和英文混杂，下面分别给出英文和中文介绍，在代码中大量采用中文注释，请有需要者自行翻译。
The introduction is mixed by Chinese and English, the following is given in English and Chinese respectively, in the code a lot of Chinese comments are used, please translate yourself if necessary.

任何用途的使用都是免费的，但是转载必须指明出处。如果在论文中使用最好有引用。
Use for any purpose is free of charge, but reproduction must indicate the source. If used in a paper it is best to have a citation.

这是一个使用python实现的粒子群算法类。共有两个文件，其中unconsoptipso.py蕴含了几种求解无约束函数的粒子群算法。consoptipso.py负责求解有约束规划问题。可同时求解整形，非整形，等式和不等式约束。（但是不建议使用其求解含等式约束的问题）

This is a class of particle swarm algorithms implemented in python. There are two files, where unconsoptipso.py contains several particle swarm algorithms for solving unconstrained functions. consoptipso.py is responsible for solving constrained planning problems. It can solve shape-shifting, non-shape-shifting, equational and inequality constraints simultaneously. (However, it is not recommended to use it for solving problems with equation constraints)

这里不给出粒子群算法的公式，给出各个参数的说明，具体使用参考给出的ipynb(jupyter notebook文件
unconsoptipso.py 共含四个类：
standardpso:标准粒子群算法
torchstdpso：标准粒子群算法采用torch张量版本，速度很慢，不建议使用 
adapwstdpso：带有自适应优化惯性权重的pso  
adapwintstdpso：自适应惯性权重的同时允许指定某一变量为整数
参数介绍：
standardpso与torchstdpso:
fitness_func：自定义的目标函数，具体定义方式请参考给出的jupyter notebook文件
lower：变量的下界，传入的是Python列表，每一维元素代表一个变量的下界
upper：变量的上界，传入的是python
dim：变量维度
sizes：粒子数量
max_v：粒子最大速度
w：惯性权重
c1：个体学习因子
c2：群体学习因子
iter_nums：最大迭代次数，超过多少次迭代停止运行
tol：忍耐度：两次迭代最优值低于多少自动退出迭代
ifplot：是否画出迭代曲线，True为画出
sovmax：是否解最大值问题，默认最小值，选择True后会将结果自动翻转（包括将ifplot图的曲线翻转），但必须预先给fitness_func乘以-1

adaptwstdpso:
自适应惯性权重，惯性权重随着迭代次数衰减，这可以使粒子群算法加快收敛速度，并且更容易使粒子群算法找到最优值。新增三个参数
wstart: 初始权重
wend:结束权重
wdecfun：衰减函数，可选f1,f2,f3,f4，其中f4有参数，f4c，越大前期衰减越快。 四种函数如下：其中f1线性，f2先慢后快，f3先快后慢。
![1](https://user-images.githubusercontent.com/92018576/201522700-ca85bd9c-ecfd-4cf1-bf40-70898995ebaa.png)


Here do not give the formula of the particle swarm algorithm, give the description of each parameter, the specific use of reference given ipynb (jupyter notebook file）
unconsoptipso.py contains a total of four classes.
standardpso:standard particle swarm algorithm
torchstdpso: the standard particle swarm algorithm uses the torch tensor version, which is very slow and not recommended 
adapwstdpso: pso with adaptive optimized inertia weights  
adapwintstdpso: adaptive inertia weights while allowing to specify a variable as an integer
Parameter description.
standardpso:
fitness_func: custom target function, please refer to the given jupyter notebook file for the definition
lower: lower bound of the variable, passed as a Python list, each dimensional element represents a lower bound of the variable
upper: the upper bound of the variable, passed in as python
dim
sizes
max_v
w
c1
c2
iter_nums
tol
ifplot
sovmax
![1](https://user-images.githubusercontent.com/92018576/201522700-ca85bd9c-ecfd-4cf1-bf40-70898995ebaa.png)
