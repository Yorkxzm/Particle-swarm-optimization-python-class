介绍由中文和英文混杂，下面分别给出英文和中文介绍，在代码中大量采用中文注释，请有需要者自行翻译。
The introduction is mixed by Chinese and English, the following is given in English and Chinese respectively, in the code a lot of Chinese comments are used, please translate yourself if necessary.

任何用途的使用都是免费的，但是转载必须指明出处。如果在论文中使用最好有引用。
Use for any purpose is free of charge, but reproduction must indicate the source. If used in a paper it is best to have a citation.

这是一个使用python实现的粒子群算法类。共有两个文件，其中unconsoptipso.py蕴含了几种求解无约束函数的粒子群算法。consoptipso.py负责求解有约束规划问题。可同时求解整形，非整形，等式和不等式约束。（但是不建议使用其求解含等式约束的问题）

This is a class of particle swarm algorithms implemented in python. There are two files, where unconsoptipso.py contains several particle swarm algorithms for solving unconstrained functions. consoptipso.py is responsible for solving constrained planning problems. It can solve shape-shifting, non-shape-shifting, equational and inequality constraints simultaneously. (However, it is not recommended to use it for solving problems with equation constraints)

这里不给出粒子群算法的公式，
unconsoptipso.py 共含四个类：standardpso（标准粒子群算法），
![1](https://user-images.githubusercontent.com/92018576/201522700-ca85bd9c-ecfd-4cf1-bf40-70898995ebaa.png)
