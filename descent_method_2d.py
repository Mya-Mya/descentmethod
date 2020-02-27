#!/usr/bin/env python
# coding: utf-8

# In[13]:


'2次元の降下法最適化を提供する。'


# In[1]:


import numpy as np


# In[3]:


class DescentMethod2D(object):
    '''
    最適化を提供する。
    '''
    def __init__(self):
        self.name='DescentMethod2D'
    def execute(self,f,w_0:np.ndarray,step:int):
        '''
        Parameters
        ----------
        f : object
            2変数関数を表すオブジェクト。関数の傾きが
            f.grad(w:np.ndarray)->np.ndarray
            によって得られる必要がある。
        w_0 : np.ndarray
            [x_0,y_0]と入れること。
        Returns
        ----x---
        w : np.ndarray
            最終的な[x,y]
        x_log : list
            全てのステップにおけるxの履歴
        y_log : list
            全てのステップにおけるyの履歴
        '''


# In[4]:


class StochasticGradientDescent(DescentMethod2D):
    '最も基本的な最急降下法。'
    def __init__(self,eta=0.01):
        '''
        eta : float
            学習率。
        '''
        self.eta=eta
        self.name='SGD η={}'.format(self.eta)
    def execute(self,f,w:np.ndarray,step:int):
        x_log=[w[0]]
        y_log=[w[1]]
        for i in range(step):
            g=f.grad(w) #今の場所の傾き
            w-=g*self.eta #傾きの逆方向へ移動
            
            x_log.append(w[0])
            y_log.append(w[1])
        return w,x_log,y_log


# In[5]:


class MomentumSGD(DescentMethod2D):
    '慣性を考慮した降下法。'
    def __init__(self,eta=0.01,gamma=0.9):
        '''
        eta : float
            学習率。
        gamma : float
            前回の慣性を記憶する割合。
        '''
        self.eta=eta
        self.gamma=gamma
        self.name='MomentumSGD η={} γ={}'.format(self.eta,self.gamma)
    def execute(self,f,w:np.ndarray,step:int):
        x_log=[w[0]]
        y_log=[w[1]]
        m=np.array([0,0])#慣性
        for i in range(step):
            g=f.grad(w) #今の場所の傾き
            m=m*self.gamma + g*self.eta #前回の慣性と傾きから今の慣性を求める
            w-=m #慣性に従って移動
            
            x_log.append(w[0])
            y_log.append(w[1])
        return w,x_log,y_log


# In[6]:


class NesterovAcceleratedGradient(DescentMethod2D):
    '慣性を考える際に一歩先を見る降下法。'
    def __init__(self,eta=0.01,gamma=0.9):
        '''
        eta : float
            学習率。
        gamma : float
            前回の慣性を記憶する割合。
        '''
        self.eta=eta
        self.gamma=gamma
        self.name='NAG η={} γ={}'.format(self.eta,self.gamma)
    def execute(self,f,w:np.ndarray,step:int):
        x_log=[w[0]]
        y_log=[w[1]]
        m=np.array([0,0])#慣性
        for i in range(step):
            g=f.grad(w-m) #ちょっと先の場所の傾き
            m=m*self.gamma + g*self.eta #前回の慣性と傾きから今の慣性を求める
            w-=m #慣性に従って移動
            
            x_log.append(w[0])
            y_log.append(w[1])
        return w,x_log,y_log


# In[7]:


class AdaGrad(DescentMethod2D):
    '変化具合に応じて学習率を変える降下法。'
    def __init__(self,eta0=0.001,epsilon=1e-8):
        '''
        eta0 : float
            学習率の初期値。
        epsilon : float
            ゼロ除算を防止するための小さな値。
        '''
        self.eta0=eta0
        self.epsilon=epsilon
        self.name='AdaGrad η0={} ε={}'.format(eta0,epsilon)
    def execute(self,f,w:np.ndarray,step:int):
        x_log=[w[0]]
        y_log=[w[1]]
        v=np.repeat(.0,2) #vの初期値
        for i in range(step):
            g=f.grad(w) #今の場所の傾き
            v+=g*g #vの更新
            w-=self.eta0/(np.sqrt(v)+self.epsilon)*g #移動 (eta/sqrt(v)+epsilon)*g
            x_log.append(w[0])
            y_log.append(w[1])
        return w,x_log,y_log


# In[8]:


class RMSprop(DescentMethod2D):
    '最近の変化具合を記憶しながら学習率を調整する降下法。'
    def __init__(self,eta0=0.01,gamma=0.99,epsilon=1e-8):
        '''
        eta0 : float
            学習率の初期値。
        gamma : float
            前回のvを記憶する割合。
        epsilon : float
            ゼロ除算を防止するための小さな値。
        '''
        self.eta0=eta0
        self.gamma=gamma
        self.one_minus_gamma=1-gamma
        self.epsilon=epsilon
        self.name='RMSprop η0={} γ={} ε={}'.format(eta0,gamma,epsilon)
    def execute(self,f,w:np.ndarray,step:int):
        x_log=[w[0]]
        y_log=[w[1]]
        v=np.repeat(0,2) #vの初期値
        for i in range(step):
            g=f.grad(w) #今の場所の傾き
            v=self.gamma*v + self.one_minus_gamma*g*g #vの更新
            w-=self.eta0/(np.sqrt(v)+self.epsilon)*g #移動
            x_log.append(w[0])
            y_log.append(w[1])
        return w,x_log,y_log


# In[9]:


class AdaDelta(DescentMethod2D):
    '学習率の単位を揃えた降下法。'
    def __init__(self,gamma=0.95,epsilon=1e-6):
        '''
        Parameters
        ----------
        gamma : float
            前回の各パラメータを記憶する割合。
        epsilon : float
            ゼロ除算を防止するための小さな値。
        '''
        self.gamma=gamma
        self.one_minus_gamma=1.-gamma
        self.epsilon=epsilon
        self.name='AdaDelta γ={} ε={}'.format(gamma,epsilon)
    def execute(self,f,w:np.ndarray,step:int):
        x_log=[w[0]]
        y_log=[w[1]]
        delta_w=np.repeat(.0,2)
        s=np.repeat(.0,2)
        v=np.repeat(.0,2)
        for i in range(step):
            g=f.grad(w) #今の場所の傾き
            v=self.gamma*v + self.one_minus_gamma*g*g
            s=self.gamma*s + self.one_minus_gamma*delta_w*delta_w
            delta_w= -g* np.sqrt( (s+self.epsilon)/(v+self.epsilon) )
            w+=delta_w
            x_log.append(w[0])
            y_log.append(w[1])
        return w,x_log,y_log


# In[10]:


class Adam(DescentMethod2D):
    'みんな大好きアダムオプティマイザ。'
    def __init__(self,eta=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
        '''
        Parameters
        ----------
        eta : float
            学習率。
        beta1 : float
            前回のmを引き継ぐ割合。
        beta2 : float
            前回のvを引き継ぐ割合。
        epsilon : float
            ゼロ除算を防止するための小さな値。
        '''
        self.eta=eta
        self.beta1=beta1
        self.one_minus_beta1=1-beta1
        self.beta2=beta2
        self.one_minus_beta2=1-beta2
        self.epsilon=epsilon
        self.name='Adam η={} β1={} β2={} ε={}'.format(eta,beta1,beta2,epsilon)
    def execute(self,f,w:np.ndarray,step:int):
        x_log=[w[0]]
        y_log=[w[1]]
        m=np.repeat(.0,2)
        v=np.repeat(.0,2)
        for i in range(step):
            g=f.grad(w)
            m= self.beta1*m + self.one_minus_beta1*g
            v= self.beta2*v + self.one_minus_beta2*g*g
            estimated_m=m/self.one_minus_beta1
            estimated_v=v/self.one_minus_beta2
            w=w-self.eta/(np.sqrt(estimated_v)+self.epsilon)*estimated_m
            x_log.append(w[0])
            y_log.append(w[1])
        return w,x_log,y_log


# In[ ]:


class SDProp(DescentMethod2D):
    'NTTの考案したもの。'
    def __init__(self,eta=0.001,gamma=0.99,epsilon=1e-8):
        '''
        Parameters
        ----------
        eta : float
            学習率。
        gamma : float
            前回のmとvを記憶する割合。
        epsilon : float
            ゼロ除算を防止するための小さな値。
        '''
        self.eta=eta
        self.gamma=gamma
        self.one_minus_gamma=1-gamma
        self.epsilon=epsilon
        self.name='SDProp η={} γ={} ε={}'.format(eta,gamma,epsilon)
    def execute(self,f,w:np.ndarray,step:int):
        x_log=[w[0]]
        y_log=[w[1]]
        m=np.repeat(.0,2)
        v=np.repeat(.0,2)
        for i in range(step):
            g=f.grad(w)
            m= self.gamma*m + self.one_minus_gamma*g
            v= self.gamma*v + self.one_minus_gamma*(g-m)**2
            estimated_v=v/(1-self.gamma**(i+1))
            w-=self.eta/(np.sqrt(estimated_v)+self.epsilon)*g
            x_log.append(w[0])
            y_log.append(w[1])
        return w,x_log,y_log


# In[ ]:


class Adastand(DescentMethod2D):
    'NTTの考案したもの。'
    def __init__(self,eta=0.001,beta1=0.7,beta2=0.99,epsilon=1e-8):
        '''
        Parameters
        ----------
        eta : float
            学習率。
        beta1 : float
            前回のmを引き継ぐ割合。
        beta2 : float
            前回のvを引き継ぐ割合。
        epsilon : float
            ゼロ除算を防止するための小さな値。
        '''
        self.eta=eta
        self.beta1=beta1
        self.one_minus_beta1=1-beta1
        self.beta2=beta2
        self.one_minus_beta2=1-beta2
        self.epsilon=epsilon
        self.name='Adastand η={} β1={} β2={} ε={}'.format(eta,beta1,beta2,epsilon)
    def execute(self,f,w:np.ndarray,step:int):
        x_log=[w[0]]
        y_log=[w[1]]
        m=np.repeat(.0,2)
        v=np.repeat(.0,2)
        for i in range(step):
            g=f.grad(w)
            m= self.beta1*m + self.one_minus_beta1*(g-m)
            v= self.beta2*v + self.one_minus_beta2*(g-m)**2
            estimated_m=m/(1-self.beta1**(i+1))
            estimated_v=v/(1-self.beta2**(i+1))
            w=w-self.eta/(np.sqrt(estimated_v)+self.epsilon)*estimated_m
            x_log.append(w[0])
            y_log.append(w[1])
        return w,x_log,y_log


# In[ ]:


class AMSGrad(DescentMethod2D):
    '傾きの情報を減衰させない。'
    def __init__(self,eta=0.001,beta1=0.9,beta2=0.99,epsilon=1e-8):
        '''
        Parameters
        ----------
        eta : float
            学習率。
        beta1 : float
            前回のmを引き継ぐ割合。
        beta2 : float
            前回のvを引き継ぐ割合。
        epsilon : float
            ゼロ除算を防止するための小さな値。
        '''
        self.eta=eta
        self.beta1=beta1
        self.one_minus_beta1=1-beta1
        self.beta2=beta2
        self.one_minus_beta2=1-beta2
        self.epsilon=epsilon
        self.name='AMSGrad η={} β1={} β2={} ε={}'.format(eta,beta1,beta2,epsilon)
    def execute(self,f,w:np.ndarray,step:int):
        x_log=[w[0]]
        y_log=[w[1]]
        m=np.repeat(.0,2)
        v=np.repeat(.0,2)
        estimated_v=np.repeat(.0,2)
        for i in range(step):
            g=f.grad(w)
            m= self.beta1*m + self.one_minus_beta1*g
            v= self.beta2*v + self.one_minus_beta2*g*g
            estimated_v=np.array([max(v[i],estimated_v[i])for i in range(2)])#max(ev[i],v[i])for all i
            w=w-self.eta/(np.sqrt(estimated_v)+self.epsilon)*m
            x_log.append(w[0])
            y_log.append(w[1])
        return w,x_log,y_log

