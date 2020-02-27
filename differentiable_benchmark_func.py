#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
testfuncs.pyとは異なり、ここでは微分を行えるようなベンチマーク関数を定義する。
'''


# In[2]:


import numpy as np


# In[3]:


class BenchmarkFunction(object):
    '''
    ベンチマーク関数 f(w) w=x1,...,xN を表す。
    NAME : 名前。
    DIM : 次元数。
    DOMAIN : 各引数の値域。[(x1_min,x1_max),...,(xN_min,xN_max)]
    MINW : 最小値を取る引数。[w1,...,wM]
    '''
    DIM=0
    DOMAIN=[]
    MINW=[]
    def __call__(self,w:np.ndarray):
        'f(w)を返す。'
        pass
    def grad(self,w:np.ndarray)->np.ndarray:
        'wにおける傾き(∂f/∂w)(w)を返す。'
        pass


# In[4]:


class Matyas(BenchmarkFunction):
    NAME='Matyas'
    DIM=2
    DOMAIN=[(-10,10),(-10,10)]
    MINW=[[-10,-10],[10,10]]
    def __call__(self,w:np.ndarray):
        x,y=w[0],w[1]
        return 0.26*(x**2+y**2)-0.48*x*y
    def grad(self,w:np.ndarray)->np.ndarray:
        x,y=w
        mx=0.52*x-0.48*y
        my=0.52*y-0.48*x
        return np.array([mx,my])


# In[5]:


class Mccormick(BenchmarkFunction):
    NAME='Mccormick'
    DIM=2
    DOMAIN=[(-1.5,4),(-3,4)]
    MINW=[[-0.547,-1.547]]
    def __call__(self,w:np.ndarray):
        x,y=w
        return np.sin(x+y)+(x-y)**2-1.5*x+2.5*y+1 
    def grad(self,w:np.ndarray)->np.ndarray:
        x,y=w
        A=np.cos(x+y)
        #mx=cos(x+y)+2x-2y-1.5
        mx=A+2*x+-2*y-1.5
        #my=cos(x+y)+2y-2x+2.5
        my=A+2*y-2*x+2.5
        return np.array([mx,my])


# In[6]:


class Himmelblau(BenchmarkFunction):
    'みんな大好き。'
    NAME='Himmelblau'
    DIM=2
    DOMAIN=[(-5,5),(-5,5)]
    MINW=[[3,2],[-2.81,3.13],[-3.78,-3.28],[3.58,-1.85]]
    def __call__(self,w:np.ndarray):
        x,y=w[0],w[1]
        return (x*x+y-11)**2+(x+y*y-7)**2
    def grad(self,w:np.ndarray)->np.ndarray:
        x,y=w[0],w[1]
        A=(x*x+y-11)*2
        B=(x+y*y-7)*2
        return np.array([A*2*x+B,A+B*2*y])


# In[7]:


class Eggcrate(BenchmarkFunction):
    NAME='Eggcrate'
    DIM=2
    DOMAIN=[(-5,5),(-5,5)]
    MINW=[[0,0]]
    def __call__(self,w:np.ndarray):
        x,y=w[0],w[1]
        return x**2+y**2+25*(np.sin(x)**2+np.sin(y)**2)
    def grad(self,w:np.ndarray)->np.ndarray:
        x,y=w[0],w[1]
        #mx=2x+25*2sinxcosx
        _2x=2*x
        mx=_2x+25*np.sin(_2x)
        #my=2y+25*2sinycosy
        _2y=2*y
        my=_2y+25*np.sin(_2y)
        print('x={} y={} mx={} my={}'.format(x,y,mx,my))
        return np.array([mx,my])


# In[8]:


class Bird(BenchmarkFunction):
    NAME='Bird'
    DIM=2
    DOMAIN=[(-10,10),(-10,10)]
    MINW=[[-6.282,6.282]]
    def __call__(self,w:np.ndarray):
        x,y=w[0],w[1]
        sinx=np.sin(x)
        cosy=np.cos(y)
        return sinx*np.exp((1-cosy)**2)+cosy*np.exp((1-sinx)**2)+(x-y)**2
    def grad(self,w:np.ndarray)->np.ndarray:
        x,y=w[0],w[1]
        sinx=np.sin(x)
        siny=np.sin(y)
        cosx=np.cos(x)
        cosy=np.cos(y)
        _1_min_sinx=1-sinx
        _1_min_cosy=1-cosy
        A=np.exp(_1_min_sinx**2)
        B=np.exp(_1_min_cosy**2)
        _2_x_min_y=2*(x-y)
        mx= cosx*B -cosy*A*2*_1_min_sinx*cosx +_2_x_min_y
        my= sinx*B*2*_1_min_cosy*sinx -siny*A -_2_x_min_y
        return np.array([mx,my])


# In[9]:


class StyblinskiTang(BenchmarkFunction):
    NAME='StyblinskiTang'
    DIM=2
    DOMAIN=[(-5,4),(-3,4)]
    MINW=[[-2.903,-2.903]]
    def __call__(self,w:np.ndarray):
        x,y=w[0],w[1]
        return (x**4-16*x**2+5*x + y**4-16*y**2+5*y)*0.5
    def grad(self,w:np.ndarray)->np.ndarray:
        return 2*w**3-16*w+2.5


# In[10]:


class Beale(BenchmarkFunction):
    NAME='Beale'
    DIM=2
    DOMAIN=[(-4.5,4.5),(-4.5,4.5)]
    MINW=[[3,0.5]]
    def __call__(self,w:np.ndarray):
        x,y=w[0],w[1]
        return (1.5-x+x*y)**2 +(2.25-x+x*y*y)**2 +(2.625-x+x*y**3)**2
    def grad(self,w:np.ndarray)->np.ndarray:
        x,y=w[0],w[1]
        gx=2*(1.5-x+x*y)*(-1+y)+2*(2.25-x+x*y*y)*(-1+y*y) +2*(2.625-x+x*y**3)*(-1+y**3)
        gy=x*( 3-2*x -2*x*y +9*y +4*x*y**3 +15.75*y*y -6*x*y*y + 6*x*y**5 )
        return np.array([gx,gy])


# In[11]:


class DeckKersaArts(BenchmarkFunction):
    '平坦な谷底が続くも最適解が隠れている。'
    NAME='DeckKersaArts'
    DIM=2
    DOMAIN=[(-20,20),(-20,20)]
    MINW=[[0,15],[0,-15]]
    def __call__(self,w:np.ndarray):
        x,y=w[0],w[1]
        A=(x*x+y*y)**2
        return x*x*1e5 +y*y -A +(1e-5)*A*A
    def grad(self,w:np.ndarray)->np.ndarray:
        x,y=w[0],w[1]
        A=(x*x+y*y)
        B=-2*A+(1e-5)*4*A**3
        gx=(1e5-B)*2*x
        gy=(1-B)*2*y
        return np.array([gx,gy])


# In[12]:


class Adjiman(BenchmarkFunction):
    'スキーのパラレル。'
    NAME='Adjiman'
    DIM=2
    DOMAIN=[(-1,26),(-1,1)]
    MINW=[[0,0]]
    def __call__(self,w:np.ndarray):
        x,y=w[0],w[1]
        return np.cos(x)*np.sin(y)-x/(y*y+1)
    def grad(self,w:np.ndarray)->np.ndarray:
        x,y=w[0],w[1]
        A=y*y+1
        gx= -np.sin(x)*np.sin(y) -1/A
        gy= np.cos(x)*np.cos(y) +2*x*y/(A*A)
        return np.array([gx,gy])


# In[13]:


class Leon(BenchmarkFunction):
    NAME='Leon'
    DIM=2
    DOMAIN=[(0,10),(0,10)]
    MINW=[[1,1]]
    def __call__(self,w:np.ndarray):
        x,y=w[0],w[1]
        return 100*(y-x**3)**2+(1-x)**2
    def grad(self,w:np.ndarray)->np.ndarray:
        x,y=w[0],w[1]
        A=y-x**3
        gx=2*(-300*A*x*x +x -1)
        gy=200*A
        return np.array([gx,gy])


# In[14]:


class Booth(BenchmarkFunction):
    NAME='Booth'
    DIM=2
    DOMAIN=[(-10,10),(-10,10)]
    MINW=[[1,3]]
    def __call__(self,w:np.ndarray):
        x,y=w[0],w[1]
        return (x+2*y-7)**2+(2*x+y-5)**2
    def grad(self,w:np.ndarray)->np.ndarray:
        x,y=w[0],w[1]
        A=2*(x+2*y-7)
        B=2*(2*x+y-5)
        return np.array([A+B+B,A+A+B])


# In[16]:


class Exponential(BenchmarkFunction):
    '縄文土器。'
    NAME='Exponential'
    DIM=2
    DOMAIN=[(-1,1),(-1,1)]
    MINW=[[0,0]]
    def __call__(self,w:np.ndarray):
        x,y=w[0],w[1]
        return -np.exp(-0.5*(x*x+y*y))
    def grad(self,w:np.ndarray)->np.ndarray:
        x,y=w[0],w[1]
        return w*np.exp(-0.5*(x*x+y*y))


# In[21]:


class Schwerel222(BenchmarkFunction):
    '薬包紙。'
    NAME='Schwerel222'
    DIM=2
    DOMAIN=[(-100,100),(-100,100)]
    MINW=[[0,0]]
    def __call__(self,w:np.ndarray):
        return np.sum(np.abs(w))*np.abs(w[0]*w[1])
    def grad(self,w:np.ndarray)->np.ndarray:
        x,y=w[0],w[1]
        A=1+np.abs(y)
        if x>0:mx=A
        elif x<0:mx=-A
        else: mx=0
        B=1+np.abs(x)
        if y>0:my=B
        elif y<0:my=-B
        else: my=0
        return np.array([mx,my])


# In[25]:


class SumSquare(BenchmarkFunction):
    'グラタンの皿。'
    NAME='SumSquare'
    DIM=2
    DOMAIN=[(-1,1),(-1,1)]
    MINW=[[0,0]]
    _grad_bias=np.array([2,4])
    def __call__(self,w:np.ndarray):
        return w[0]**2+2*w[1]**2
    def grad(self,w:np.ndarray)->np.ndarray:
        return self._grad_bias*w

