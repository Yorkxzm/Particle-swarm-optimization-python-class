import numpy as np
import matplotlib.pyplot as plt
#标准粒子群优化（不能添加约束条件，只能给定变量范围）
#注意NP数组可以像dataframe那样索引！
#标准粒子群优化（不能添加约束条件，只能给定变量范围）
#注意NP数组可以像dataframe那样索引！
#标准粒子群优化（不能添加约束条件，只能给定变量范围）
#注意NP数组可以像dataframe那样索引！
class standardpso:
    def __init__(self,fitness_func,lower,upper,dim,sizes,max_v=0.6,w=1,c1=2,c2=2,iter_nums=500,tol=1e-10,ifplot=True,sovmax=False):
        self.fitness_func=fitness_func
        self.lower=lower
        self.upper=upper
        self.dim=dim
        self.sizes=sizes
        self.w=w
        self.c1=c1
        self.c2=c2
        self.iter_num=iter_nums
        self.ifplot=ifplot
        self.sovmax=sovmax
        self.max_v=max_v
        self.tol=tol
    def vupdate(self,V,X,pbest,gbest):#优化速度，输入一个数组
        #max v是速度的最大值
        size=V.shape[0]#粒子数量
        r1=np.random.random((size,1))
        r2=np.random.random((size,1))
        V=self.w*V+self.c1*r1*(pbest-X)+self.c2*r2*(gbest-X)
        #注意这一步gbest-x,gbest是一个1*dim的数组，显然和X不同维度。这里用到了numpy矩阵运算的性质简化了公式，本来gbest-X是要写for循环一列一列减的
        #但是numpy直接可以用gbest+-X，意思是先取-X,再用gbest逐行加，最后得到一个和X同维度的东西
        V[V<-self.max_v]=-self.max_v
        V[V>self.max_v]=self.max_v
        return V
    #更新X
    def xupdate(self,X,V):#优化位置
        #X(k+1)=X(k)+tv(k+1)(t=1)
        return X+V
    def fit(self):#这个func是目标函数
        tol2=100#待更新忍耐度
        fitness_value_list=[]#记录种群最优适应度的变化，这个是用来画图的
        size=self.sizes
        X=np.zeros((size,self.dim))
        #初始化X
        for i in range(0,self.dim):
            X[:,i]=np.random.uniform(self.lower[i],self.upper[i],size=(size))
        V=np.random.uniform(-self.max_v,self.max_v,size=(size,self.dim))
        #第一步操作用于确认pbest和gbest,放到外面
        p_fitness=self.fitness_func(X)#这个返回的是np数组
        g_fitness=p_fitness.min()#获取最小值
        #这俩保存最优值
        fitness_value_list.append(g_fitness)#群体最优值被认为是当前最优值
        pbest=X
        gbest=X[p_fitness.argmin()]#argmin是寻找最小函数值对应的变量，可以使用它在X中获取索引
        #上面的操作是用来初始化pbest和gbest的
        for i in range(1,self.iter_num):
            V=self.vupdate(V,X,pbest,gbest)
            X=self.xupdate(X,V)
            for dimsa in range(self.dim):#变量范围约束
                X1=X[:,dimsa]#这里直接取行
                 #也就是说如果采用索引搜索，和原来的变量占用一个空间
                X1[X1>self.upper[dimsa]]=self.upper[dimsa]#类似于dataframe的搜索操作
                X1[X1<self.lower[dimsa]]=self.lower[dimsa]
            p_fitness2=self.fitness_func(X)#这个返回的是np数组
            g_fitness2=p_fitness2.min()#获取最小值，1*1
            #更新最优位置
            for j in range(size):
                if p_fitness[j]>p_fitness2[j]:#P_fitness是函数值，是个一维数组，长为size
                    p_fitness[j]=p_fitness2[j]
                    pbest[j]=X[j]#更新种群最优位置，pbest[j]是位置best。索引与p_fitness对应，从新的X中获取
            if g_fitness>g_fitness2:#群体最优值出现了
                gbest=X[p_fitness2.argmin()]#从群体最优值中获取索引位置
                tol2=g_fitness-g_fitness2
                g_fitness=g_fitness2
            fitness_value_list.append(g_fitness)
            if tol2<self.tol:
                break#两次寻优低于忍耐度，break
        if self.sovmax==True:
            fitness_value_list=-1*np.array(fitness_value_list)
        self.besty=fitness_value_list[-1]
        self.gbest=gbest
        self.fitness_value_list=fitness_value_list
        if self.ifplot==True:
            #画图
            plt.rcParams['font.family'] = ['sans-serif']#防止中文报错
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
            plt.plot(fitness_value_list)
            plt.title('迭代过程')
    def getresult(self):
        return self.gbest,self.besty
    def printresult(self):
        print(f"最优变量:{self.gbest}")
        print("最优值是:%5f" % self.besty)
    def getvaluelist(self):
        return self.fitness_value_list

class adapwstdpso:
    def __init__(self,fitness_func,lower,upper,dim,sizes,max_v=0.6,w=1,wstart=2,wend=0.4,wdecfun='f3',f4c=2,c1=2,c2=2,iter_nums=500,tol=1e-10,ifplot=True,sovmax=False):
        self.fitness_func=fitness_func
        self.lower=lower
        self.upper=upper
        self.dim=dim
        self.sizes=sizes
        self.w=w
        self.c1=c1
        self.c2=c2
        self.iter_num=iter_nums
        self.ifplot=ifplot
        self.sovmax=sovmax
        self.max_v=max_v
        self.tol=tol
        self.wstart=wstart
        self.wend=wend
        self.wdecfun=wdecfun
        self.f4c=f4c
    def vupdate(self,V,X,pbest,gbest):#优化速度，输入一个数组
        #max v是速度的最大值
        size=V.shape[0]#粒子数量
        r1=np.random.random((size,1))
        r2=np.random.random((size,1))
        V=self.w*V+self.c1*r1*(pbest-X)+self.c2*r2*(gbest-X)
        #注意这一步gbest-x,gbest是一个1*dim的数组，显然和X不同维度。这里用到了numpy矩阵运算的性质简化了公式，本来gbest-X是要写for循环一列一列减的
        #但是numpy直接可以用gbest+-X，意思是先取-X,再用gbest逐行加，最后得到一个和X同维度的东西
        V[V<-self.max_v]=-self.max_v
        V[V>self.max_v]=self.max_v
        return V
    #更新X
    def xupdate(self,X,V):#优化位置
        #X(k+1)=X(k)+tv(k+1)(t=1)
        return X+V
    def wupdate(self,k):
        #
        if self.wdecfun=='f1':#linear dec
            w=self.wend+(self.wstart-self.wend)*(self.iter_num-k)*(1/self.iter_num)
            return w
        elif self.wdecfun=='f2':
            w=self.wstart-((self.wstart-self.wend)*((k/self.iter_num)**2))
            return w
        elif self.wdecfun=='f3':
            w=self.wstart-((self.wstart-self.wend)*((2*k/self.iter_num-((k/self.iter_num)**2))))
            return w
        elif self.wdecfun=='f4':
            w=self.wend*(self.wstart/self.wend)**(1/(1+self.f4c*k/self.iter_num))
            #f4c是衰减速度，f4c越大前期衰减越快
            return w   
    def fit(self):#这个func是目标函数
        tol2=100#待更新忍耐度
        self.wlist=[]
        fitness_value_list=[]#记录种群最优适应度的变化，这个是用来画图的
        size=self.sizes
        X=np.zeros((size,self.dim))
        #初始化X
        for i in range(0,self.dim):
            X[:,i]=np.random.uniform(self.lower[i],self.upper[i],size=(size))
        V=np.random.uniform(-self.max_v,self.max_v,size=(size,self.dim))
        #第一步操作用于确认pbest和gbest,放到外面
        p_fitness=self.fitness_func(X)#这个返回的是np数组
        g_fitness=p_fitness.min()#获取最小值
        #这俩保存最优值
        fitness_value_list.append(g_fitness)#群体最优值被认为是当前最优值
        pbest=X
        gbest=X[p_fitness.argmin()]#argmin是寻找最小函数值对应的变量，可以使用它在X中获取索引
        #上面的操作是用来初始化pbest和gbest的
        for i in range(1,self.iter_num):
            V=self.vupdate(V,X,pbest,gbest)
            X=self.xupdate(X,V)
            for dimsa in range(self.dim):#变量范围约束
                X1=X[:,dimsa]#这里直接取行
                 #也就是说如果采用索引搜索，和原来的变量占用一个空间
                X1[X1>self.upper[dimsa]]=self.upper[dimsa]#类似于dataframe的搜索操作
                X1[X1<self.lower[dimsa]]=self.lower[dimsa]
            p_fitness2=self.fitness_func(X)#这个返回的是np数组
            g_fitness2=p_fitness2.min()#获取最小值，1*1
            #更新最优位置
            for j in range(size):
                if p_fitness[j]>p_fitness2[j]:#P_fitness是函数值，是个一维数组，长为size
                    p_fitness[j]=p_fitness2[j]
                    pbest[j]=X[j]#更新种群最优位置，pbest[j]是位置best。索引与p_fitness对应，从新的X中获取
            if g_fitness>g_fitness2:#群体最优值出现了
                gbest=X[p_fitness2.argmin()]#从群体最优值中获取索引位置
                tol2=g_fitness-g_fitness2
                g_fitness=g_fitness2
            self.w=self.wupdate(i)#更新w
            fitness_value_list.append(g_fitness)
            self.wlist.append(self.w)
            if tol2<self.tol:
                break#两次寻优低于忍耐度，break
        if self.sovmax==True:
            fitness_value_list=-1*np.array(fitness_value_list)
        self.besty=fitness_value_list[-1]
        self.gbest=gbest
        self.fitness_value_list=fitness_value_list
        if self.ifplot==True:
            #画图
            plt.rcParams['font.family'] = ['sans-serif']#防止中文报错
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
            plt.plot(fitness_value_list)
            plt.title('迭代过程')
    def getresult(self):
        return self.gbest,self.besty
    def printresult(self):
        print(f"最优变量:{self.gbest}")
        print("最优值是:%5f" % self.besty)
    def getvaluelist(self):
        return self.fitness_value_list
    def getwlist(self):
        return self.wlist

class adapwintstdpso:
    def __init__(self,fitness_func,lower,upper,dim,sizes,isint,max_v=0.6,w=1,wstart=2,wend=0.4,wdecfun='f3',f4c=2,c1=2,c2=2,iter_nums=500,tol=1e-10,ifplot=True,sovmax=False):
        self.fitness_func=fitness_func
        self.lower=lower
        self.upper=upper
        self.dim=dim
        self.sizes=sizes
        self.w=w
        self.c1=c1
        self.c2=c2
        self.iter_num=iter_nums
        self.ifplot=ifplot
        self.sovmax=sovmax
        self.max_v=max_v
        self.tol=tol
        self.wstart=wstart
        self.wend=wend
        self.wdecfun=wdecfun
        self.f4c=f4c
        self.isint=isint
    def vupdate(self,V,X,pbest,gbest):#优化速度，输入一个数组
        #max v是速度的最大值
        size=V.shape[0]#粒子数量
        r1=np.random.random((size,1))
        r2=np.random.random((size,1))
        V=self.w*V+self.c1*r1*(pbest-X)+self.c2*r2*(gbest-X)
        #注意这一步gbest-x,gbest是一个1*dim的数组，显然和X不同维度。这里用到了numpy矩阵运算的性质简化了公式，本来gbest-X是要写for循环一列一列减的
        #但是numpy直接可以用gbest+-X，意思是先取-X,再用gbest逐行加，最后得到一个和X同维度的东西
        V[V<-self.max_v]=-self.max_v
        V[V>self.max_v]=self.max_v
        return V
    #更新X
    def xupdate(self,X,V):#优化位置
        #X(k+1)=X(k)+tv(k+1)(t=1)
        return X+V
    def wupdate(self,k):
        #
        if self.wdecfun=='f1':#linear dec
            w=self.wend+(self.wstart-self.wend)*(self.iter_num-k)*(1/self.iter_num)
            return w
        elif self.wdecfun=='f2':
            w=self.wstart-((self.wstart-self.wend)*((k/self.iter_num)**2))
            return w
        elif self.wdecfun=='f3':
            w=self.wstart-((self.wstart-self.wend)*((2*k/self.iter_num-((k/self.iter_num)**2))))
            return w
        elif self.wdecfun=='f4':
            w=self.wend*(self.wstart/self.wend)**(1/(1+self.f4c*k/self.iter_num))
            #f4c是衰减速度，f4c越大前期衰减越快
            return w   
    def fit(self):#这个func是目标函数
        tol2=100#待更新忍耐度
        self.wlist=[]
        fitness_value_list=[]#记录种群最优适应度的变化，这个是用来画图的
        size=self.sizes
        X=np.zeros((size,self.dim))
        #初始化X
        for i in range(0,self.dim):
            X[:,i]=np.random.uniform(self.lower[i],self.upper[i],size=(size))
        V=np.random.uniform(-self.max_v,self.max_v,size=(size,self.dim))
        #第一步操作用于确认pbest和gbest,放到外面
        p_fitness=self.fitness_func(X)#这个返回的是np数组
        g_fitness=p_fitness.min()#获取最小值
        #这俩保存最优值
        fitness_value_list.append(g_fitness)#群体最优值被认为是当前最优值
        pbest=X
        gbest=X[p_fitness.argmin()]#argmin是寻找最小函数值对应的变量，可以使用它在X中获取索引
        #上面的操作是用来初始化pbest和gbest的
        for i in range(1,self.iter_num):
            V=self.vupdate(V,X,pbest,gbest)
            X=self.xupdate(X,V)
            for dimsa in range(self.dim):#变量范围约束
                X1=X[:,dimsa]#这里直接取行
                 #也就是说如果采用索引搜索，和原来的变量占用一个空间
                X1[X1>self.upper[dimsa]]=self.upper[dimsa]#类似于dataframe的搜索操作
                X1[X1<self.lower[dimsa]]=self.lower[dimsa]
            for dimsa in range(self.dim):#变量范围约束
                if self.isint[dimsa]==1:#是整数
                    X[:,dimsa]=np.rint(X[:,dimsa])#取整
            p_fitness2=self.fitness_func(X)#这个返回的是np数组
            g_fitness2=p_fitness2.min()#获取最小值，1*1
            #更新最优位置
            for j in range(size):
                if p_fitness[j]>p_fitness2[j]:#P_fitness是函数值，是个一维数组，长为size
                    p_fitness[j]=p_fitness2[j]
                    pbest[j]=X[j]#更新种群最优位置，pbest[j]是位置best。索引与p_fitness对应，从新的X中获取
            if g_fitness>g_fitness2:#群体最优值出现了
                gbest=X[p_fitness2.argmin()]#从群体最优值中获取索引位置
                tol2=g_fitness-g_fitness2
                g_fitness=g_fitness2
            self.w=self.wupdate(i)#更新w
            fitness_value_list.append(g_fitness)
            self.wlist.append(self.w)
            if tol2<self.tol:
                break#两次寻优低于忍耐度，break
        if self.sovmax==True:
            fitness_value_list=-1*np.array(fitness_value_list)
        self.besty=fitness_value_list[-1]
        self.gbest=gbest
        self.fitness_value_list=fitness_value_list
        if self.ifplot==True:
            #画图
            plt.rcParams['font.family'] = ['sans-serif']#防止中文报错
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
            plt.plot(fitness_value_list)
            plt.title('迭代过程')
    def getresult(self):
        return self.gbest,self.besty
    def printresult(self):
        print(f"最优变量:{self.gbest}")
        print("最优值是:%5f" % self.besty)
    def getvaluelist(self):
        return self.fitness_value_list
    def getwlist(self):
        return self.wlist

import torch#可由torch加速的，但是不知道为什么不好用
class torchstdpso:
    def __init__(self,fitness_func,lower,upper,dim,sizes,max_v=0.6,w=1,c1=2,c2=2,iter_nums=500,tol=1e-10,ifplot=True,sovmax=False,cudatrue=False):
        self.fitness_func=fitness_func
        self.lower=lower
        self.upper=upper
        self.dim=dim
        self.sizes=sizes
        self.w=w
        self.c1=c1
        self.c2=c2
        self.iter_num=iter_nums
        self.ifplot=ifplot
        self.sovmax=sovmax
        self.max_v=max_v
        self.tol=tol
        self.cudatrue=cudatrue
    def vupdate(self,V,X,pbest,gbest,r1,r2):#优化速度，输入一个数组
        #max v是速度的最大值
        V=self.w*V+self.c1*r1*(pbest-X)+self.c2*r2*(gbest-X)
        #注意这一步gbest-x,gbest是一个1*dim的数组，显然和X不同维度。这里用到了numpy矩阵运算的性质简化了公式，本来gbest-X是要写for循环一列一列减的
        #但是numpy直接可以用gbest+-X，意思是先取-X,再用gbest逐行加，最后得到一个和X同维度的东西
        V[V<-self.max_v]=-self.max_v
        V[V>self.max_v]=self.max_v
        return V
    #更新X
    def xupdate(self,X,V):#优化位置
        #X(k+1)=X(k)+tv(k+1)(t=1)
        return X+V
    def fit(self):#这个func是目标函数
        tol2=100#待更新忍耐度
        device = torch.device("cuda") if self.cudatrue==True else torch.device("cpu")
        fitness_value_list=[]#记录种群最优适应度的变化，这个是用来画图的
        size=self.sizes
        X=torch.Tensor(size,self.dim).to(device)
        #初始化X
        #torch.Tensor(size,dim).uniform_(-1,1)代替np.random.uniform(self.lower[i],self.upper[i],size=(size))
        #注意是.Tensor不是tensor
        for i in range(0,self.dim):
            X[:,i]=torch.Tensor(size).uniform_(self.lower[i],self.upper[i]).to(device)
        V=torch.Tensor(size,self.dim).uniform_(-self.max_v,self.max_v).to(device)
        #第一步操作用于确认pbest和gbest,放到外面
        p_fitness=self.fitness_func(X)#这个返回的是np数组
        g_fitness=p_fitness.min()#获取最小值
        #这俩保存最优值
        fitness_value_list.append(g_fitness)#群体最优值被认为是当前最优值
        pbest=X
        gbest=X[p_fitness.argmin()]#argmin是寻找最小函数值对应的变量，可以使用它在X中获取索引
        #上面的操作是用来初始化pbest和gbest的
        for i in range(1,self.iter_num):
            size=V.shape[0]#粒子数量
            r1=torch.rand((size,1)).to(device)
            r2=torch.rand((size,1)).to(device)
            V=self.vupdate(V,X,pbest,gbest,r1,r2)
            X=self.xupdate(X,V)
            for dimsa in range(self.dim):#变量范围约束
                X1=X[:,dimsa]# #也就是说如果采用索引搜索，和原来的变量占用一个空间
               
                X1[X1>self.upper[dimsa]]=self.upper[dimsa]#类似于dataframe的搜索操作
                X1[X1<self.lower[dimsa]]=self.lower[dimsa]
            p_fitness2=self.fitness_func(X)#这个返回的是np数组
            g_fitness2=p_fitness2.min()#获取最小值，1*1
            #更新最优位置
            for j in range(size):
                if p_fitness[j]>p_fitness2[j]:#P_fitness是函数值，是个一维数组，长为size
                    p_fitness[j]=p_fitness2[j]
                    pbest[j]=X[j]#更新种群最优位置，pbest[j]是位置best。索引与p_fitness对应，从新的X中获取
            if g_fitness>g_fitness2:#群体最优值出现了
                gbest=X[p_fitness2.argmin()]#从群体最优值中获取索引位置
                tol2=g_fitness-g_fitness2
                g_fitness=g_fitness2
            fitness_value_list.append(g_fitness)
            if tol2<self.tol:
                break#两次寻优低于忍耐度，break
        if self.sovmax==True:
            fitness_value_list=-1*np.array(fitness_value_list)
        self.besty=fitness_value_list[-1]
        self.gbest=gbest
        self.fitness_value_list=fitness_value_list
        if self.ifplot==True:
            #画图
            plt.rcParams['font.family'] = ['sans-serif']#防止中文报错
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
            plt.plot(fitness_value_list)
            plt.title('迭代过程')
    def getresult(self):
        return self.gbest,self.besty
    def printresult(self):
        print(f"最优变量:{self.gbest}")
        print("最优值是:%5f" % self.besty)
    def getvaluelist(self):
        return self.fitness_value_list