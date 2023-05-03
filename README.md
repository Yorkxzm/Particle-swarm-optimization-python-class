# <p align="center">PSOclasses</p>

介绍由中文和英文混杂，下面分别给出英文和中文介绍，在代码中大量采用中文注释，请有需要者自行翻译。<br>
The introduction is mixed by Chinese and English, the following is given in English and Chinese respectively, in the code a lot of Chinese comments are used, please translate yourself if necessary.<br>

任何用途的使用都是免费的，但是转载必须指明出处。如果在自己的论文中使用了代码，建议附上引用（可以直接引用网址）。<br>
Use for any purpose is free of charge, but reproduction must indicate the source. If used in a paper it is best to have a citation.

这是一个使用python实现的粒子群算法类。共有两个文件，其中unconsoptipso.py蕴含了几种求解无约束函数的粒子群算法。consoptipso.py负责求解有约束规划问题。可同时求解整形，非整形，等式和不等式约束。（但是不建议使用其求解含等式约束的问题）<br>

This is a class of particle swarm algorithms implemented in python. There are two files, where unconsoptipso.py contains several particle swarm algorithms for solving unconstrained functions. consoptipso.py is responsible for solving constrained planning problems. It can solve shape-shifting, non-shape-shifting, equational and inequality constraints simultaneously. (However, it is not recommended to use it for solving problems with equation constraints)<br>

共含三个使用教学jupyter notebook文档，分别求解了无约束函数优化，有约束函数优化和函数拟合问题。接下来提供中英双语的参数和文件介绍<br>
A total of three tutorial jupyter notebook documents are included, solving unconstrained function optimization, constrained function optimization, and function fitting problems respectively.The introduction of parameters and documentation are provided in both English and Chinese below<br>


这里不给出粒子群算法的公式，给出各个参数的说明，具体使用参考给出的ipynb(jupyter notebook文件<br>
共含四个类：<br>
standardpso:标准粒子群算法<br>
torchstdpso：标准粒子群算法采用torch张量版本，速度很慢，不建议使用 <br>
adapwstdpso：带有自适应优化惯性权重的pso  <br>
adapwintstdpso：自适应惯性权重的同时允许指定某一变量为整数<br>
参数介绍：<br>
standardpso与torchstdpso:<br>
fitness_func：自定义的目标函数，具体定义方式请参考给出的jupyter notebook文件<br>
lower：变量的下界，传入的是Python列表，每一维元素代表一个变量的下界<br>
upper：变量的上界，传入的是python列表，每一维元素代表一个变量的上界<br>
dim：变量维度<br>
sizes：粒子数量<br>
max_v：粒子最大速度<br>
w：惯性权重<br>
c1：个体学习因子<br>
c2：群体学习因子<br>
iter_nums：最大迭代次数，超过多少次迭代停止运行<br>
tol：忍耐度：两次迭代最优值低于多少自动退出迭代<br>
ifplot：是否画出迭代曲线，True为画出<br>
sovmax：是否解最大值问题，默认最小值，选择True后会将结果自动翻转（包括将ifplot图的曲线翻转），但必须预先给fitness_func乘以-1<br>
<br>
adaptwstdpso:<br>
自适应惯性权重，惯性权重随着迭代次数衰减，这可以使粒子群算法加快收敛速度，并且更容易使粒子群算法找到最优值。新增三个参数<br>
wstart: 初始权重<br>
wend:结束权重<br>
wdecfun：衰减函数，可选f1,f2,f3,f4，其中f4有参数，f4c，越大前期衰减越快。 四种函数如下：其中f1线性，f2先慢后快，f3先快后慢。<br>
![1](https://user-images.githubusercontent.com/92018576/201522700-ca85bd9c-ecfd-4cf1-bf40-70898995ebaa.png)
其中F3函数效果一般较好，所以设置为默认值。<br>
adapwintstdpso:可以指定变量为整形<br>
传入参数isint是一个python列表，每一维元素代表某个变量是否为整形，0为非整形，1为整形
如[0,0,1]代表x3是整数变量<br>

我们求解了一个无约束最大化和一个最小化问题，全部都是较为复杂的带有多个局部极值点的函数，见ipynb。<br>


consoptipso.py共两个类，这两个类全部有惯性权重自适应。<br>
这些类的使用需要传入参数aeq和a，其基础思想为设置罚函数。<br>
可以自由定义aeq和a,加上判断即可。具体的使用方式见jupyter notebook说明，建议参考其写法。<br>

consadawpsoaeq：<br>
只能添加不等式约束，其效果较好，但对于某些复杂的问题仍然需要多次运行和调参。<br>
我们求解了三个优化问题：<br>
包括：<br>
![image](https://user-images.githubusercontent.com/92018576/201523798-b0afe9af-4475-465c-ae0e-560fcd4a5c97.png)<br>
典型的非凸优化（最大化），当x为整型变量，y为整型变量，两者都为连续变量时分别有最优值12,12.25,12.5<br>
![image](https://user-images.githubusercontent.com/92018576/201524333-6e51697e-ce26-4b30-bb28-0c115cfb2375.png)<br>
![image](https://user-images.githubusercontent.com/92018576/201524480-c1364448-3273-4a91-a3ac-e12035843ab7.png)<br>
consadawpso（不推荐使用）：<br>
可以同时添加等式和不等式约束，鉴于粒子找到等式约束（空间上的直线）条件是很困难的，因此我们定义tol，当边界值小于tol时被认为满足要求。
求解了一个带有等式约束的优化问题（最大化）：<br>
![image](https://user-images.githubusercontent.com/92018576/201524004-40a9c64c-230a-4b35-8158-3000ea7dd0d7.png)<br>


The formula of particle swarm optimization algorithm is not given here, and the description of each parameter is given. The specific use of ipynb is given in the reference.

unconsoptipso.py has four classes:

standardpso: Standard PSO <br>

torchstdpso: The torch tensor version of the standard PSO is very slow and not recommended

adapwstdpso: pso with Adaptively optimized Inertia Weight <br>

adapwintstdpso: Adaptive inertia weight while allowing to specify a variable as an integer <br>

Parameter: <br>

standardpso vs. torchstdpso:

fitness_func: a custom target function, as defined in the given jupyter notebook <br>

lower: The lower bound of a variable, passed a Python list, where each element represents a lower bound <br>

upper: The upper bound of the variable,passed a Python list, where each element represents a upper bound <br>

dim: variable dimension <br>

sizes: number of particles <br>

max_v: Maximum particle velocity <br>

w: Inertia weight <br>

c1: Individual learning factor <br>

c2: Group learning factor <br>

iter_nums: maximum number of iterations after which to stop <br>

tol: Tolerance: The optimal value of two iterations below how much automatically quit the iteration <br>

ifplot: Whether to plot the iteration curve, True = <br>

sovmax: Whether to solve the maximum value problem, the default minimum value, the result will be automatically inverted (including inverting the curve of ifplot) if selected, but fitness_func must be multiplied by -1<br>

<br>

adaptwstdpso:<br>

The adaptive inertia weight, the inertia weight decays with the number of iterations, which can make the PSO converge faster, and it is easier for the PSO to find the optimal value. Three new <br> parameters

wstart: initial weight <br>

wend: end weight <br>

wdecfun: Decay function, optional f1,f2,f3,f4, where f4 has parameters, f4c, the larger the decay is faster in the early stage. The four functions are as follows: where f1 is linear, f2 is slow and then fast, and f3 is fast and then slow. <br>
![1](https://user-images.githubusercontent.com/92018576/201522700-ca85bd9c-ecfd-4cf1-bf40-70898995ebaa.png)
The F3 function generally works better, so it is set to the default value. <br>

adapwintstdpso: Variable can be specified as integer <br>

The isint argument is a python list with each dimension representing whether the variable is an integer, 0 for non-integer, and 1 for integer

For example, [0,0,1] means that x3 is an integer variable.

We solved an unconstrained maximization and a minimization problem, both of which were complex functions with multiple local extremes.See ipynb . <br>

consoptipso.py has two classes, both of which have inertia weight adaptation. <br>

The use of these classes requires the parameters aeq and a, and the basic idea is to set a penalty function. <br>

It is free to define aeq and a, plus the judgment. See the jupyter notebook for details, and it is recommended to refer to its writing method. <br>



consadawpsoaeq: <br>

Only inequality constraints can be added, which works well, but still requires multiple runs and parameter tuning for some complex problems. <br>

We solve three optimization problems: <br>

Includes: <br>

![image](https://user-images.githubusercontent.com/92018576/201524630-95c5959c-d249-44cf-8ebf-1c74194c354c.png)<br>

Typical non-convex optimization (maximization), where x is an integer variable, y is an integer variable, and both are continuous variables. 12,12.25,12.5<br>
![image](https://user-images.githubusercontent.com/92018576/201524638-2bce1836-5e56-44ed-b4e6-33f1c47a3c40.png)<br>

![image](https://user-images.githubusercontent.com/92018576/201524639-b5f040e4-d433-4359-b642-6f232f4e4636.png)<br>

consadawpso（Not recommended）: <br>

It is possible to add both equality and inequality constraints. Since it is difficult for a particle to find the equality constraint (a hyper-line in space), we define tol, which is considered satisfied when the boundary value is less than tol.

Solve an optimization problem (maximization) with equality constraints: <br>
![image](https://user-images.githubusercontent.com/92018576/201524642-e070faa3-81ba-4795-ad33-a4221f3c7b5b.png)<br>
