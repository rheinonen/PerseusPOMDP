import numpy as np
import scipy.special as sc
from datetime import datetime
import random
import math
#scipy.special.cython_special as spec

class SimpleEnv2D: ### MAIN ENV CLASS FOR 2D OLFACTORY SEARCH
    def __init__(self,params):
        self.seed=datetime.now()
        random.seed(self.seed)
        self.t=0
        self.dims=params["dims"] # number of (x,y) gridpoints. Ny should be odd
        self.Lx=params["Lx"] #downwind box size
        self.Ly=params["Ly"] #crosswind box size
        self.dx=self.Lx/(self.dims[0]-1)
        self.dy=self.Ly/(self.dims[1]-1)
        self.x0=params["x0"] #integer downwind position of source relative to lefthand boundary
        self.xarr=np.linspace(0,self.Lx,self.dims[0])
        self.yarr=np.linspace(0,self.Ly,self.dims[1])
        self.D=params["D"] #turbulent diffusivity
        self.agent_size=params["agent_size"]
        self.tau=params["tau"] #particle lifetime
        self.V=params["V"] #mean flow speed
        self.R=params["R"] #emission rate
        self.dt=params["dt"] #time step
        self.actions=[np.array([-1,0]),np.array([1,0]),np.array([0,-1]),np.array([0,1])]
        self.numobs=params['max_detections']+2
        self.obs=[i for i in range(-1,params['max_detections']+1)]
        self.gamma=params["gamma"] #discount rate
        self.numactions=4
        penalty=params["exit_penalty"]
        self.easy_likelihood=params["easy_likelihood"]
        self.Uoverv=params["Uoverv"]
        self.agent=None
        self.y0=params['y0']
        self.pos=np.array([self.x0,self.y0])
        time_reward=params['time_reward']
        self.shaped=False
        if "2d" in params:
            self.twod=params["2d"]
        else:
            self.twod=False
        #if self.twod:
        #    import scipy.special.cython_special as spec
        self.shaping_factor=params["shaping_factor"]
        if self.shaping_factor=="q":
            self.shaped=True
        elif self.shaping_factor>0:
            self.shaped=True
        self.shaping_power=params["shaping_power"]
        self.entropy_factor=params["entropy_factor"]
        if self.entropy_factor>0:
            self.shaped=True



        dist=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        for i in range(-(self.dims[0]-1),self.dims[0]):
            for j in range(-(self.dims[1]-1),self.dims[1]):
                dist[i,j]=np.abs(i)+np.abs(j)
        #if time_reward:
        #    self.qvalue=-dist
        #else:
        #    self.qvalue=self.gamma**dist

        self.dist=dist
        self.likelihood=[]
        x=np.arange(-self.dims[0]+1,self.dims[0])
        y=np.arange(-self.dims[1]+1,self.dims[1])
        x=x[:,None]
        y=y[None,:]
        for i in range(self.obs[-1]+1):
            l=self.compute_likelihood(x,y,i)
            assert(l[self.dims[0]-1,self.dims[1]-1]==0)
            self.likelihood.append(l)

        shaping=np.zeros((self.numactions,2*self.dims[0]-1,2*self.dims[1]-1))
        shapefunc=None
        if self.shaped:
            if self.shaping_factor=="q":
                shapefunc=1/self.gamma-self.gamma**(self.dist-1)
            elif self.shaping_power==0:
                shapefunc=np.log(1+self.dist)
            else:
                shapefunc=self.shaping_factor*self.dist**self.shaping_power
            if self.entropy_factor>0:
                shapefunc=shapefunc+self.entropy_factor*(self.likelihood*np.log(self.likelihood,out=np.zeros_like(self.likelihood),where=self.likelihood!=0)+(1-self.likelihood)*np.log(1-self.likelihood,out=np.zeros_like(self.likelihood),where=self.likelihood!=1))
        self.shapefunc=shapefunc

        if self.shaped:
            for a in range(self.numactions):
                action=self.actions[a]
                shaping[a,:,:]=self.gamma*np.roll(shapefunc,tuple(-action),axis=(0,1))
                shaping[a,0,0]=self.gamma*shapefunc[0,0]
            shaping=shapefunc[None,:,:]-shaping
        self.shaping=shaping

        if not time_reward:
            self.reward0=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
            self.reward1=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
            self.reward2=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
            self.reward3=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
            self.reward4=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))


            self.reward1[1,0]=1
            self.reward2[-1,0]=1
            self.reward3[0,1]=1
            self.reward4[0,-1]=1

            self.reward0[self.dims[0]-self.x0:self.dims[0],:]=-penalty
            self.reward1[self.dims[0]-1-self.x0:self.dims[0],:]=-penalty

            self.reward0[-(self.dims[0]):-self.x0,:]=-penalty
            self.reward2[-(self.dims[0]-1):-self.x0+1,:]=-penalty

            self.reward0[:,self.dims[1]-self.y0:self.dims[1]]=-penalty
            self.reward3[:,self.dims[1]-self.y0-1:self.dims[1]]=-penalty

            self.reward0[:,-(self.dims[1]):-self.y0]=-penalty
            self.reward4[:,-(self.dims[1]-1):-self.y0+1]=-penalty
            self.rewards=[self.reward1+shaping[0,:,:],self.reward2+shaping[1,:,:],self.reward3+shaping[2,:,:],self.reward4+shaping[3,:,:]]
        else:
            rew=-1*np.ones((2*self.dims[0]-1,2*self.dims[1]-1))
            rew[0,0]=0
            self.rewards=[rew,rew,rew,rew,rew]

    def change_shaping_factor(self,new_factor,vf):
        shaping=np.zeros((self.numactions,2*self.dims[0]-1,2*self.dims[1]-1))
        vf.shift_value((new_factor-self.shaping_factor)*self.shapefunc)
        self.shaping_factor=new_factor
        for a in range(self.numactions):
            action=self.actions[a]
            tmp0=np.roll(self.likelihood*self.shapefunc,tuple(-action),axis=(0,1))
            tmp0[-action[0],-action[1]]=0
            tmp0[0,0]=0
            p=1-self.likelihood
            p[0,0]=0
            tmp1=np.roll(p*self.shapefunc,tuple(-action),axis=(0,1))
            tmp1[-action[0],-action[1]]=0
            tmp1[0,0]=0
            shaping[a,:,:]=self.gamma*(tmp0+tmp1)
        shaping=self.shapefunc[None,:,:]-shaping
        shaping=shaping*self.shaping_factor
        self.rewards=[self.reward1+shaping[0,:,:],self.reward2+shaping[1,:,:],self.reward3+shaping[2,:,:],self.reward4+shaping[3,:,:]]


    def stepInTime(self):
        self.t+=1

    def reset(self):
        self.t=0

    def set_pos(self,x0,y0):
        self.x0=x0
        self.y0=y0

    def set_agent(self,agent):
        self.agent=agent


    def get_likelihood(self,x,y,n):
        return self.likelihood[n][x+self.dims[0]-1,y+self.dims[1]-1]

    def compute_likelihood(self,x,y,n):
        if self.twod:
            r=np.sqrt((self.dx*x)**2+(self.dy*y)**2)
            r1=r+(r==0)
            ell=np.sqrt(self.D*self.tau/(1+self.V**2*self.tau/(4*self.D)))
            tmp=sc.kn(0,r1/ell)
            tmp*=self.R*self.dt/np.log(ell/self.agent_size)*np.exp(self.dx*x*self.V/2/self.D)
            if n==self.obs[-1]:
                out =0
                for i in range(self.obs[-1]):
                    out=out+np.exp(-tmp)*tmp**i/math.factorial(i)
                return (1-out)*(r>0)
            else:
                return np.exp(-tmp)*tmp**n/math.factorial(n)*(r>0)
        if self.easy_likelihood:
            if isinstance(x,int):
                x=np.array([x],dtype='float64')
                y=np.array([y],dtype='float64')
            else:
                x=x.astype('float64')
                y=y.astype('float64')
            return 1/np.cosh(self.dy*self.Uoverv/self.dx*y*np.divide(np.ones_like(x),x,out=np.zeros_like(x),where=x!=0))**2*(x>0)
        r=np.sqrt((self.dx*x)**2+(self.dy*y)**2)
        r1=r+(r==0)
        tmp=self.dt*self.agent_size*self.R/r1
        tmp*=np.exp(-r1/np.sqrt(self.D*self.tau/(1+self.V**2*self.tau/(4*self.D))))
        tmp*=np.exp(self.dx*x*self.V/(2*self.D))
        if n==self.obs[-1]:
            out=0
            for i in range(self.obs[-1]):
                out=out+np.exp(-tmp)*tmp**i/math.factorial(i)
            return (1-out)*(r>0)
        else:
            return np.exp(-tmp)*tmp**n/math.factorial(n)*(r>0)

    def get_rate(self,x,y):
        if self.twod:
            r=np.sqrt((self.dx*x)**2+(self.dy*y)**2)
            r1=r+(r==0)
            ell=np.sqrt(self.D*self.tau/(1+self.V**2*self.tau/(4*self.D)))
            tmp=sc.kn(0,r1/ell)
            tmp*=self.R/np.log(ell/self.agent_size)*np.exp(self.dx*x*self.V/2/self.D)
            return tmp*(r>0)

        r=np.sqrt((self.dx*x)**2+(self.dy*y)**2)
        tmp=np.divide(self.dt*self.agent_size*self.R, r, out=np.zeros_like(r), where=r!=0)
        tmp*=np.exp(-r/np.sqrt(self.D*self.tau/(1+self.V**2*self.tau/(4*self.D))))
        tmp*=np.exp(self.dx*x*self.V/(2*self.D))
        return tmp

    def transition_function(self,belief,action):
        b=np.roll(belief,tuple(action),axis=(0,1))
        if np.array_equal(action,[1,0]):
            tsrc=belief[0,0]+belief[-1,0]
        elif np.array_equal(action,[-1,0]):
            tsrc=belief[0,0]+belief[1,0]
        elif np.array_equal(action,[0,1]):
            tsrc=belief[0,0]+belief[0,-1]
        elif np.array_equal(action,[0,-1]):
            tsrc=belief[0,0]+belief[0,1]
        else:
            raise RuntimeError("not a valid action")
        tsrc_out=np.zeros_like(b)
        tsrc_out[0,0]=1

        ps=[]
        taus=[]
        ps.append(tsrc)
        taus.append(tsrc_out)

        for obs in range(self.numobs-1):
            tmp=np.roll(self.likelihood[obs],(self.dims[0],self.dims[1]),axis=(0,1))*b
            tmp[0,0]=0
            ps.append(np.sum(tmp))
            taus.append(tmp/np.sum(tmp))
        return ps,taus

    def get_g(self,alpha,action): #first element associated with detection, second non-detection
        out=[]
        g_src=np.zeros_like(alpha)
        g_src[0,0]=alpha[0,0]
        g_src[-action[0],-action[1]]=alpha[0,0]
        out.append(g_src)
        for obs in range(self.numobs-1):
            g=alpha*np.roll(self.likelihood[obs],(self.dims[0],self.dims[1]),axis=(0,1))
            g=np.roll(g,tuple(-action),axis=(0,1))
            g[0,0]=0
            g[-action[0],-action[1]]=0
            out.append(g.copy())
        return out

    def getObs(self,pos):
        n=np.random.poisson(self.get_rate(pos[0]-self.x0,pos[1]-self.y0))
        if n>self.obs[-1]:
            n=self.obs[-1]
        #print(x,l)
        return n

    def transition(self,pos,action):
        tmp=pos+action
        if not self.outOfBounds(tmp):
            return tmp # agent is unmoved if attempts to leave simulation bounds
        return pos

    def outOfBounds(self,pos):
        if (pos[0]<0 or pos[0]>self.dims[0]-1 or pos[1] < 0 or pos[1] > self.dims[1]-1):
            return True
        return False

    def getReward(self,true_pos,action):
        r=true_pos-self.pos
        # if (action==np.array([0,0])).all():
        #     return self.rewards[0][r[0],r[1]]
        if (action==np.array([1,0])).all():
            return self.rewards[0][r[0],r[1]]
        elif (action==np.array([-1,0])).all():
            return self.rewards[1][r[0],r[1]]
        elif (action==np.array([0,1])).all():
            return self.rewards[2][r[0],r[1]]
        elif (action==np.array([0,-1])).all():
            return self.rewards[3][r[0],r[1]]

class BinaryDiscrimination2D:
    def __init__(self,params):
        self.seed=datetime.now()
        random.seed(self.seed)
        self.t=0
        self.dims=params["dims"] # number of (x,y) gridpoints. Ny should be odd
        self.Lx=params["Lx"] #downwind box size
        self.Ly=params["Ly"] #crosswind box size
        self.dx=self.Lx/(self.dims[0]-1)
        self.dy=self.Ly/(self.dims[1]-1)
        self.x0=params["x0"] #integer downwind position of source relative to lefthand boundary
        self.xarr=np.linspace(0,self.Lx,self.dims[0])
        self.yarr=np.linspace(0,self.Ly,self.dims[1])
        self.D=params["D"] #turbulent diffusivity
        self.agent_size=params["agent_size"]
        self.tau=params["tau"] #particle lifetime
        self.V=params["V"] #mean flow speed
        self.R=params["R"] #emission rate
        self.dt=params["dt"] #time step
        self.actions=["east","west","north","south","abort"]
        self.numobs=4
        self.obs=[True,False,"source","aborted"]
        self.gamma=params["gamma"] #discount rate
        self.numactions=5
        rew=params["rew"]
        self.rew=rew
        self.shaped=False
        self.shaping_factor=params["shaping_factor"]

        if self.shaping_factor>0:
            self.shaped=True


        self.agent=None
        self.y0=params['y0']
        self.pos=np.array([self.x0,self.y0])
        self.s1=np.array(params['low_separation'])
        self.s2=np.array(params['high_separation'])
        self.true_sep=params['true_sep']

        dist=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        for i in range(-(self.dims[0]-1),self.dims[0]):
            for j in range(-(self.dims[1]-1),self.dims[1]):
                dist[i,j]=np.abs(i)+np.abs(j)

        self.dist=dist

        self.likelihood=np.zeros((2,2*self.dims[0]-1,2*self.dims[1]-1))
        for i in range(-(self.dims[0]-1),self.dims[0]):
            for j in range(-(self.dims[1]-1),self.dims[1]):
                    #self.likelihood[i,j]=1/np.cosh(np.abs(self.yarr[j])/self.xarr[i]/self.intens)**2*sc.exp1(self.intens**2*self.xarr[i]**2/self.c0)
                self.likelihood[0,i,j]=self.get_likelihood(i,j,sep=0)
                self.likelihood[1,i,j]=self.get_likelihood(i,j,sep=1)

        if int(self.s1[0]/2)==self.s1[0]/2 and int(self.s1[1]/2)==self.s1[1]/2:
            self.likelihood[0,int(self.s1[0]/2),int(self.s1[1]/2)]==1
            self.likelihood[0,int(-self.s1[0]/2),int(-self.s1[1]/2)]==1
        #if int(self.s2[0]/2)==self.s2[0]/2 and int(self.s2[1]/2)==self.s2[1]/2:
        #    self.likelihood[1,int(self.s2[0]/2),int(self.s2[1]/2)]==1
        #    self.likelihood[1,int(-self.s2[0]/2),int(-self.s2[1]/2)]==1

        shaping=np.zeros((self.numactions,2,2*self.dims[0]-1,2*self.dims[1]-1))
        #shapefunc=np.stack([self.dist,self.dist],axis=0)
        #shapefunc[1,0,0]=np.max(self.dist)+1
        #if self.shaped:
        #    for a in range(self.numactions):
        #        action=self.actions[a]
        #        if action=="abort":
        #            shaping[a,:,:,:]=self.gamma*shapefunc[1,0,0]
        #            continue
        #        action=self.str_to_array(action)
        #        tmp0=np.roll(self.likelihood*shapefunc,tuple(-action),axis=(1,2))
        #        tmp0[:,-action[0],-action[1]]=0
        #        tmp0[:,0,0]=0
        #        p=1-self.likelihood
        #        p[:,0,0]=0
        #        tmp1=np.roll(p*shapefunc,tuple(-action),axis=(1,2))
        #        tmp1[:,-action[0],-action[1]]=0
        ##        tmp1[:,0,0]=0
         #       tmp2=np.zeros_like(tmp1)
         #       tmp2[1,0,0]=shapefunc[1,0,0]
         #       shaping[a,:,:,:]=self.gamma*(tmp0+tmp1+tmp2)
         #   shaping=shapefunc[None,:,:,:]-shaping
          #  shaping=shaping*self.shaping_factor




        #penalty=params["exit_penalty"]
        #self.easy_likelihood=params["easy_likelihood"]
        #self.Uoverv=params["Uoverv"

        reward0=np.zeros((2,)+(2*self.dims[0]-1,2*self.dims[1]-1))
        reward1=-1*np.ones((2,)+(2*self.dims[0]-1,2*self.dims[1]-1))
        reward2=-1*np.ones((2,)+(2*self.dims[0]-1,2*self.dims[1]-1))
        reward3=-1*np.ones((2,)+(2*self.dims[0]-1,2*self.dims[1]-1))
        reward4=-1*np.ones((2,)+(2*self.dims[0]-1,2*self.dims[1]-1))
        reward1[0,-1,0]=rew
        reward2[0,1,0]=rew
        reward3[0,0,-1]=rew
        reward4[0,0,1]=rew
        reward0[:,0,0]=0
        reward1[:,0,0]=0
        reward2[:,0,0]=0
        reward3[:,0,0]=0
        reward4[:,0,0]=0
        self.rewards=[reward1+shaping[0,:,:,:],reward2+shaping[1,:,:,:],reward3+shaping[2,:,:,:],reward4+shaping[3,:,:,:],reward0+shaping[4,:,:,:]]

        # x=np.arange(self.dims[0])
        # x=x[:,None]
        # y=np.arange(self.dims[1])
        # y=y[None,:]



    def get_likelihood(self,x,y,sep=0):
        if sep==0:
            r1=np.sqrt((self.dx*(x-self.s1[0]/2))**2+(self.dy*(y-self.s1[1]/2))**2)
            r1=r1+(r1==0)
            tmp1=0.5*self.dt*self.agent_size*self.R/r1
            tmp1*=np.exp(-r1/np.sqrt(self.D*self.tau/(1+self.V**2*self.tau/(4*self.D))))
            tmp1*=np.exp(self.dx*(x-self.s1[0]/2)*self.V/(2*self.D))
            r2=np.sqrt((self.dx*(x+self.s1[0]/2))**2+(self.dy*(y+self.s1[1]/2))**2)
            r2=r2+(r2==0)
            tmp2=0.5*self.dt*self.agent_size*self.R/r2
            tmp2*=np.exp(-r2/np.sqrt(self.D*self.tau/(1+self.V**2*self.tau/(4*self.D))))
            tmp2*=np.exp(self.dx*(x+self.s1[0]/2)*self.V/(2*self.D))
            return 1-np.exp(-tmp1-tmp2)
        if sep==1:
            r1=np.sqrt((self.dx*(x-self.s2[0]/2))**2+(self.dy*(y-self.s2[1]/2))**2)
            r1=r1+(r1==0)*999999
            tmp1=0.5*self.dt*self.agent_size*self.R/r1
            tmp1*=np.exp(-r1/np.sqrt(self.D*self.tau/(1+self.V**2*self.tau/(4*self.D))))
            tmp1*=np.exp(self.dx*(x-self.s2[0]/2)*self.V/(2*self.D))
            r2=np.sqrt((self.dx*(x+self.s2[0]/2))**2+(self.dy*(y+self.s2[1]/2))**2)
            r2=r2+(r2==0)*999999
            tmp2=0.5*self.dt*self.agent_size*self.R/r2
            tmp2*=np.exp(-r2/np.sqrt(self.D*self.tau/(1+self.V**2*self.tau/(4*self.D))))
            tmp2*=np.exp(self.dx*(x+self.s2[0]/2)*self.V/(2*self.D))
            return 1-np.exp(-tmp1-tmp2)

    def stepInTime(self):
        self.t+=1

    def reset(self,true_sep=None):
        self.t=0
        if true_sep is not None:
            self.true_sep=true_sep

    def set_pos(self,x0,y0,):
        self.x0=x0
        self.y0=y0

    def set_agent(self,agent):
        self.agent=agent

    def str_to_array(self,action):
        if action=="east":
            return np.array([1,0])
        if action=="west":
            return np.array([-1,0])
        if action=="north":
            return np.array([0,1])
        if action=="south":
            return np.array([0,-1])

        if action=="null":
            return np.array([0,0])
        return action


    def array_to_str(self,action):
        if np.array_equal(action,[1,0]):
            return "east"
        if np.array_equal(action,[-1,0]):
            return "west"
        if np.array_equal(action,[0,1]):
            return "north"
        if np.array_equal(action,[0,-1]):
            return "south"


    def transition_function(self,belief,action):
        if action=="abort":
            out=np.zeros((2,2*self.dims[0]-1,2*self.dims[1]-1))
            t4=np.zeros((2,2*self.dims[0]-1,2*self.dims[1]-1))
            t4[1,0,0]=1
            return [0,0,0,1],[out,out,out,t4]

        l=self.likelihood
        b=np.roll(belief,tuple(self.str_to_array(action)),axis=(1,2))
        t1=b*l
        t2=(1-l)*b
        t1[:,0,0]=0
        t2[:,0,0]=0
        #t3=np.zero_like(l)
        state=-self.str_to_array(action)
        t3=belief[0,0,0]+belief[0,state[0],state[1]]+belief[1,state[0],state[1]]
        t3_out=np.zeros((2,2*self.dims[0]-1,2*self.dims[1]-1))
        t3_out[0,0,0]=1
        t4_out=np.zeros((2,2*self.dims[0]-1,2*self.dims[1]-1))
        t4_out[1,0,0]=1

        return [np.sum(t1),np.sum(t2),t3,belief[1,0,0]],[t1/np.sum(t1),t2/np.sum(t2),t3_out,t4_out]

    def get_g(self,alpha,action): #first element associated with detection, second non-detection
        if action=="abort":
            g1=np.zeros_like(alpha)
            out4=np.ones_like(alpha)*alpha[1,0,0]
            out4[0,0,0]=0
            return[g1,g1,g1,out4]

        l=self.likelihood
        g1=l*alpha
        g2=(1-l)*alpha
        out3=np.zeros_like(g2)
        out3[0,0,0]=alpha[0,0,0]
        a=-self.str_to_array(action)
        out1=np.roll(g1,tuple(a),axis=(1,2))
        out2=np.roll(g2,tuple(a),axis=(1,2))
        out1[:,0,0]=0
        out1[:,a[0],a[1]]=0
        out2[:,0,0]=0
        out3[0,a[0],a[1]]=alpha[0,0,0]
        out4=np.zeros_like(g2)
        out4[1,0,0]=alpha[1,0,0]
        out4[1,a[0],a[1]]=alpha[1,0,0]
        return[out1,out2,out3,out4]

    def getObs(self,pos):
        if np.array_equal(pos,[self.x0,self.y0]):
            return "source"
        l=self.get_likelihood(pos[0]-self.x0,pos[1]-self.y0,sep=self.true_sep)
        x=random.random()
        return x<l

    def transition(self,pos,action):
        if action=="abort":
            return np.array([0,0])
        tmp=pos+self.str_to_array(action)
        if not self.outOfBounds(tmp):
            return tmp # agent is unmoved if attempts to leave simulation bounds
        return pos

    def outOfBounds(self,pos):
        if (pos[0]<0 or pos[0]>self.dims[0]-1 or pos[1] < 0 or pos[1] > self.dims[1]-1):
            return True
        return False


    def getReward(self,true_pos,action):
        r=true_pos-self.pos
        if action=="east":
            return self.rewards[0][self.true_sep,r[0],r[1]]
        elif action=="west":
            return self.rewards[1][self.true_sep,r[0],r[1]]
        elif action=="north":
            return self.rewards[2][self.true_sep,r[0],r[1]]
        elif action=="south":
            return self.rewards[3][self.true_sep,r[0],r[1]]
        elif action=="abort":
            return self.rewards[4][self.true_sep,r[0],r[1]]

class DNSEnv2D(SimpleEnv2D):
    def __init__(self,params):
        self.conc_data=params['data']
        self.param_a=params['param_a']
        self.param_b=params['param_b']
        self.vfrac=params['vfrac']
        self.threshold=params['threshold']
        self.t=0
        self.data_r0=None
        self.dims=params["dims"] # number of (x,y) gridpoints. Ny should be odd
        self.Lx=params["Lx"] #downwind box size
        self.Ly=params["Ly"] #crosswind box size
        self.dx=self.Lx/(self.dims[0]-1)
        self.dy=self.Ly/(self.dims[1]-1)
        self.x0=params["x0"] #integer downwind position of source relative to lefthand boundary
        self.y0=params["y0"]
        self.actions=[np.array([1,0]),np.array([-1,0]),np.array([0,1]),np.array([0,-1])]
        self.numobs=2 #only implementing a single threshold for now
        self.numactions=4
        self.obs=[i for i in range(-1,2)]
        self.dummy=params['dummy']
        self.xarr=np.linspace(0,self.Lx,self.dims[0])
        self.yarr=np.linspace(0,self.Ly,self.dims[1])
        self.pos=np.array([self.x0,self.y0])
        self.gamma=params['gamma']
        self.shaped=False
        self.shaping_factor=params["shaping_factor"]
        if self.shaping_factor=="q":
            self.shaped=True
        elif self.shaping_factor>0:
            self.shaped=True
        self.shaping_power=params["shaping_power"]
        self.entropy_factor=params["entropy_factor"]
        if self.entropy_factor>0:
            self.shaped=True
        self.likelihood=[]
        self.agent=None
        dist=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        for i in range(-(self.dims[0]-1),self.dims[0]):
            for j in range(-(self.dims[1]-1),self.dims[1]):
                dist[i,j]=np.abs(i)+np.abs(j)
        self.dist=dist
        x=np.arange(-self.dims[0]+1,self.dims[0])
        y=np.arange(-self.dims[1]+1,self.dims[1])
        x=x[:,None]
        y=y[None,:]
        for i in range(self.obs[-1]+1):
            l=self.compute_likelihood(x,y,i)
            assert(l[self.dims[0]-1,self.dims[1]-1]==0)
            self.likelihood.append(l)

        shaping=np.zeros((self.numactions,2*self.dims[0]-1,2*self.dims[1]-1))
        shapefunc=None
        if self.shaped:
            if self.shaping_factor=="q":
                shapefunc=1/self.gamma-self.gamma**(self.dist-1)
            elif self.shaping_power==0:
                shapefunc=np.log(1+self.dist)
            else:
                shapefunc=self.shaping_factor*self.dist**self.shaping_power
            if self.entropy_factor>0:
                shapefunc=shapefunc+self.entropy_factor*(self.likelihood*np.log(self.likelihood,out=np.zeros_like(self.likelihood),where=self.likelihood!=0)+(1-self.likelihood)*np.log(1-self.likelihood,out=np.zeros_like(self.likelihood),where=self.likelihood!=1))
        self.shapefunc=shapefunc

        if self.shaped:
            for a in range(self.numactions):
                action=self.actions[a]
                shaping[a,:,:]=self.gamma*np.roll(shapefunc,tuple(-action),axis=(0,1))
                shaping[a,0,0]=self.gamma*shapefunc[0,0]
            shaping=shapefunc[None,:,:]-shaping
        self.shaping=shaping

        self.reward1=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        self.reward2=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        self.reward3=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        self.reward4=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))

        self.reward1[-1,0]=1
        self.reward2[1,0]=1
        self.reward3[0,-1]=1
        self.reward4[0,1]=1

        self.rewards=[self.reward1+shaping[0,:,:],self.reward2+shaping[1,:,:],self.reward3+shaping[2,:,:],self.reward4+shaping[3,:,:]]
    def get_rate(self,x,y):
        raise NotImplementedError()

    def getObs(self,pos):
        if self.dummy:
            l=np.array(self.likelihood)
            p=l[:,pos[0]-self.x0+self.dims[0]-1,pos[1]-self.y0+self.dims[1]-1]
            return np.random.choice(len(p),p=p)
        data_pos=self.data_r0+pos-np.array([self.x0,self.y0])
        if data_pos[0]<0 or data_pos[1]<0 or data_pos[0]>=self.dims[0] or data_pos[1]>=self.dims[1]:
            return 0
        return int(self.conc_data[self.t,data_pos[0],data_pos[1]]>=self.threshold)

    def compute_likelihood(self,x,y,n):
        positive=x<0
        not_origin=x**2+y**2>0
        chi=np.divide(self.param_a,np.abs(self.dx*x),where=x!=0,out=np.zeros_like(x,dtype='float'))*np.exp(-(np.divide(y*self.dy,x*self.dx,where=x!=0,out=np.zeros_like(x*y,dtype='float'))/self.vfrac)**2)*np.heaviside(-x,0)
        C=np.divide(self.param_b,np.abs(self.dx*x),where=x!=0,out=np.zeros_like(x,dtype='float'))
        if n==0:
            return np.exp(-chi*sc.exp1(1/C))*positive+(1-positive)*not_origin
        if n==1:
            return (1-np.exp(-chi*sc.exp1(1/C)))*positive

    def set_data(self,data,data_r0=[93,16]):
        self.conc_data=data
        self.data_r0=data_r0

class DNSCorrEnv2D(SimpleEnv2D):
    def __init__(self,params):
        self.conc_data=params['data']
        self.threshold=params['threshold']
        self.t=0
        self.data_r0=None
        self.dims=params["dims"] # number of (x,y) gridpoints. Ny should be odd
        self.Lx=params["Lx"] #downwind box size
        self.Ly=params["Ly"] #crosswind box size
        self.dx=self.Lx/(self.dims[0]-1)
        self.dy=self.Ly/(self.dims[1]-1)
        self.x0=params["x0"] #integer downwind position of source relative to lefthand boundary
        self.y0=params["y0"]
        self.actions=[np.array([1,0]),np.array([-1,0]),np.array([0,1]),np.array([0,-1])]
        self.numobs=2 #only implementing a single threshold for now
        self.numactions=4
        self.obs=[i for i in range(-1,2)]
        self.xarr=np.linspace(0,self.Lx,self.dims[0])
        self.yarr=np.linspace(0,self.Ly,self.dims[1])
        self.pos=np.array([self.x0,self.y0])
        self.gamma=params['gamma']
        self.likelihood=None
        self.agent=None
        self.unconditional_likelihood=None
        self.dummy=params['dummy']

        dist=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        for i in range(-(self.dims[0]-1),self.dims[0]):
            for j in range(-(self.dims[1]-1),self.dims[1]):
                dist[i,j]=np.abs(i)+np.abs(j)
        self.dist=dist

        self.reward1=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        self.reward2=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        self.reward3=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        self.reward4=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))

        self.reward1[-1,0]=1
        self.reward2[1,0]=1
        self.reward3[0,-1]=1
        self.reward4[0,1]=1

        self.rewards=[self.reward1,self.reward2,self.reward3,self.reward4]

    def set_likelihood(self,l0,l1,l_un):
        #l0 and l1 are lists of likelihood (of detection) matrices for each action, given an observational state.
        #given as function of FINAL position (AFTER the action)
        out0=[]
        out1=[]
        xs=np.arange(self.dims[0]-1-self.x0,2*self.dims[0]-1-self.x0)
        ys=np.arange(self.dims[1]-1-self.y0,2*self.dims[1]-1-self.y0)
        #print(xs.shape,ys.shape)
        for a in range(self.numactions):
            l0_out=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
            l1_out=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
            l0_out[xs[:,None],ys[None,:]]=l0[a]
            l1_out[xs[:,None],ys[None,:]]=l1[a]
            out0.append(l0_out.copy())
            out1.append(l1_out.copy())
        #    out0[0,0]=0
        #    out1[0,0]=0
        self.likelihood=[out0,out1]
        out_un=np.zeros((2*self.dims[0]-1,2*self.dims[1]-1))
        out_un[xs[:,None],ys[None,:]]=l_un
        #out_un0[0,0]=0
        #out_un1[0,0]=0

        self.unconditional_likelihood=out_un


    def get_rate(self,x,y):
        raise NotImplementedError()

    def getObs(self,pos,ag=None):
        if self.dummy:
            l=self.get_likelihood(pos[0]-self.x0,pos[1]-self.y0,1,ag.last_obs,ag.last_action)
         #   print("prob of obs is",l)
            return int(random.random()<l)

        data_pos=self.data_r0+pos-np.array([self.x0,self.y0])
        if data_pos[0]<0 or data_pos[1]<0 or data_pos[0]>=self.dims[0] or data_pos[1]>=self.dims[1]:
            return 0
        return int(self.conc_data[self.t,data_pos[0],data_pos[1]]>=self.threshold)

    def set_data(self,data,data_r0=[93,16]):
        self.conc_data=data
        self.data_r0=data_r0

    def array_to_int(self,action):
        if np.array_equal(action,[1,0]):
            return 0
        if np.array_equal(action,[-1,0]):
            return 1
        if np.array_equal(action,[0,1]):
            return 2
        if np.array_equal(action,[0,-1]):
            return 3

    def get_likelihood(self,x,y,obs,obs_state=None,action=None):
        if action is None:
            l=self.unconditional_likelihood[x+self.dims[0]-1,y+self.dims[1]-1]
        else:
            l=self.likelihood[obs_state][self.array_to_int(action)][x+self.dims[0]-1,y+self.dims[1]-1]
        if obs==0:
            return 1-l
        else:
            return l

class TigerGrid:
    def __init__(self,gamma=0.95):
        self.seed=datetime.now()
        random.seed(self.seed)
        self.t=0
        self.numactions=3
        self.actions=[0,1,2] # forward, rotate right, rotate left
        # state is encoded by an integer 0-32 given by square index*4+direction (0 up, 1 right, 2 down, 3 left)
        self.gamma=gamma
        self.dims=(33,)
        self.numobs=5 #0 is nothing. 1 is wall. 2 is tiger. 3 is tiger+wall. 4 is reward

        reward0=np.zeros((33,))
        reward1=np.zeros((33,))
        reward2=np.zeros((33,))
        reward0[5]=100
        reward0[7]=-100
        reward0[9]=-100
        reward0[11]=100
        reward0[16]=-100
        reward0[28]=-100
        reward0[0]=-100
        reward0[3]=-100
        reward0[12]=-100
        reward1[0]=-100
        reward1[1]=-100
        reward1[2]=-100
        reward1[3]=-100
        reward2[0]=-100
        reward2[1]=-100
        reward2[2]=-100
        reward2[3]=-100
        reward1[12]=-100
        reward1[13]=-100
        reward1[14]=-100
        reward1[15]=-100
        reward2[12]=-100
        reward2[13]=-100
        reward2[14]=-100
        reward2[15]=-100
        self.rewards=[reward0,reward1,reward2]
        self.pobs=np.zeros((5,33))
        for i in range(33):
            j=self.getObs(i)
            self.pobs[j,i]=1
        self.ptrans=np.zeros((3,33,33)) # action, s, s'
        for a in range(3):
            for s in range(32):
                sp=self.transition(s,a)
                self.ptrans[a,s,sp]=1
        self.ptrans[:,-1,:-1]=1/32


    def stepInTime(self):
        self.t+=1

    def transition(self,state,action):
        if state==32:
            return random.randrange(32)
        direction=state%4
        pos=state//4
        if action==0:
            if self.facingWall(state):
                return state
            elif direction==0:
                pos=pos-4
            elif direction==1:
                if pos==1:
                    return 32
                pos=pos+1
            elif direction==2:
                pos=pos+4
            elif direction==3:
                if pos==2:
                    return 32
                pos=pos-1

        else:
            if action==1:
                direction=(direction+1)%4
            if action==2:
                direction=(direction-1)%4
        return pos*4+direction

    def getObs(self,state):
        if state==32:
            return 4
        direction=state%4
        pos=state//4
        wall=self.facingWall(state)
        if (pos==0 or pos==3) and wall:
            return 3
        if pos==0 or pos==3:
            return 2
        if wall:
            return 1
        return 0

    def getReward(self,action):
        return self.rewards[action]

    def get_g(self,alpha,action):
        ptrans=np.squeeze(self.ptrans[action,:,:])
        g=np.einsum('ij,kj,j->ki',ptrans,self.pobs,alpha)
        out=[]
        for i in range(5):
            out.append(np.squeeze(g[i,:]))
        return out

    def facingWall(self,state):
        direction=state%4
        pos=state//4
        if pos<4 and direction==0:
            return True
        if pos>=4 and direction==2:
            return True
        if (pos==0 or pos==4) and direction==3:
            return True
        if (pos==3 or pos==7) and direction==1:
            return True
        if pos==5 and direction==1:
            return True
        if pos==6 and direction==3:
            return True
        return False
#
class BernoulliBandits:
    def __init__(self,probs,Np=51,gamma=0.9):
        self.seed=datetime.now()
        random.seed(self.seed)
        self.agent=None
        self.t=0
        self.numobs=2
        self.gamma=gamma
        #self.rewards=rewards
        self.p_grid=np.linspace(0,1,Np)
        self.probs=probs # list specifying probablities of success for each arm
        self.numactions=len(probs)
        self.dims=(Np,)*self.numactions
        self.actions=[a for a in range(0,self.numactions) ]
        self.rewards=[]
        for a in range(0,self.numactions):
            x=self.p_grid
            x=np.broadcast_to(x,self.dims)
            perm=[i for i in range(0,self.numactions-1)]
            perm.insert(a,self.numactions-1)
            perm=tuple(perm)
            x=np.transpose(x,perm)
            self.rewards.append(x)

    def stepInTime(self):
        self.t+=1

    def set_agent(self,agent):
        self.agent=agent

    def getReward(self):
        #EXPECTED rewards based on current belief
        #not clear this function will be used
        return self.agent.alphas/(self.agent.alphas+self.agent.betas)

    def getObs(self,action):
        return random.random()<self.probs[action]

    def get_g(self,alpha,action):
        shape=(1,)*action+self.p_grid.shape+(1,)*(self.numactions-action-1)
        x=np.reshape(self.p_grid,shape)
        return [x*alpha,(1-x)*alpha]

class BernoulliBanditsOriginal:
    def __init__(self,probs,Np=51,gamma=0.9):
        self.seed=datetime.now()
        random.seed(self.seed)
        self.agent=None
        self.t=0
        self.numobs=2
        self.gamma=gamma
        #self.rewards=rewards
        self.p_grid=np.linspace(0,1,Np)
        self.probs=probs # list specifying probablities of success for each arm
        self.numactions=len(probs)

        self.dims=(self.numactions,Np)
        self.actions=[a for a in range(0,self.numactions) ]
        self.rewards=[]
        for a in range(0,self.numactions):
            x=np.zeros(self.dims)
            x[a,:]=self.p_grid
            self.rewards.append(x.copy())
         # expected rewards are just the probabilities of success. this structure ensures b dot r is expected reward


    def stepInTime(self):
        self.t+=1

    def set_agent(self,agent):
        self.agent=agent

    def getReward(self):
        #EXPECTED rewards based on current belief
        #not clear this function will be used
        return self.agent.alphas/(self.agent.alphas+self.agent.betas)

    def getObs(self,action):
        return random.random()<self.probs[action]

    def get_g(self,alpha,action):
        #success
        return [self.p_grid[None,:]*alpha,(1-self.p_grid[None,:])*alpha]

class Tag:
    def __init__(self,gamma=0.95):
        self.seed=datetime.now()
        random.seed(self.seed)
        self.t=0
        self.numactions=5
        self.actions=['n','s','e','w','tag']

        self.gamma=gamma
        self.dims=(29,30)
        self.numobs=2
        reward_tag=-10*np.ones(self.dims)
        for i in range(29):
            reward_tag[i,i]=10
        self.rewards=[-1*np.ones(self.dims),-1*np.ones(self.dims),-1*np.ones(self.dims),-1*np.ones(self.dims),reward_tag]
        p_trans=np.zeros((5,29,30,29,30))

        for a in range(5):
            for s11 in range(29):
                for s12 in range(30):
                    for s21 in range(29):
                        for s22 in range(30):
                            p_trans[a,s11,s12,s21,s22]=self.p_trans(self.actions[a],(s11,s12),(s21,s22))

        for a in range(5):
            for s11 in range(29):
                for s12 in range(30):
                    #print(a,s11,s12)
                    #print(np.sum(p_trans[a,s11,s12,:,:]))
                    assert(np.sum(p_trans[a,s11,s12,:,:])==1)

        self.pt=p_trans
        self.p_obs=[np.zeros((29,30)),np.zeros((29,30))]
        for i in range(29):
            for j in range(30):
                if i==j:
                    self.p_obs[1][i,j]=1
        self.p_obs[0]=1-self.p_obs[1]
        self.p_obs[0][:,29]=0

        self.pos=random.randrange(29)

    def p_trans(self,action,state1,state2):
        if state1[1]==29:
            if state2[1]==29 and state1[0]==state2[0]:
                return 1
            return 0
        if action=='tag':
            if state1[0]==state2[0]:
                if state1[0]==state1[1]:
                    if state2[1]==29:
                        return 1
                else:
                    return self.p_run(state1,state2[1])
            return 0
        if state2[0]==self.chase(state1[0],action):
            return self.p_run(state1,state2[1])
        return 0

    def transition_function(self,belief,action):
        if action=='n':
            a=0
        if action=='s':
            a=1
        if action=='e':
            a=2
        if action=='w':
            a=3
        if action=='tag':
            a=4
        t0=np.einsum('ij,ijlm,lm->lm',belief,np.squeeze(self.pt[a,:,:,:,:]),self.p_obs[0])
        t1=np.einsum('ij,ijlm,lm->lm',belief,np.squeeze(self.pt[a,:,:,:,:]),self.p_obs[1])
        if np.sum(t0)==0:
            t0_out=np.zeros_like(t0)
        else:
            t0_out=t0/np.sum(t0)
        if np.sum(t1)==0:
            t1_out=np.zeros_like(t1)
        else:
            t1_out=t1/np.sum(t1)

        return [np.sum(t0),np.sum(t1)],[t0_out,t1_out]

    def p_run(self,state,s2):
        if s2==state[1]:
            return 0.2
        actions=[]
        if self.x_coord(state[0])<=self.x_coord(state[1]):
            actions.append('e')
        if self.x_coord(state[0])>=self.x_coord(state[1]):
            actions.append('w')
        if self.y_coord(state[0])<=self.y_coord(state[1]):
            actions.append('n')
        if self.y_coord(state[0])>=self.y_coord(state[1]):
            actions.append('s')

        good_actions=[]
        for a in actions:
            if not self.out_of_bounds(state[1],a):
                good_actions.append(a)
        if not good_actions:
            actions=['e','w','n','s']
            for a in actions:
                if not self.out_of_bounds(state[1],a):
                    good_actions.append(a)
        num=len(good_actions)
        for a in good_actions:
            if s2==self.chase(state[1],a):
                return 1./num*0.8
        return 0


    def transition(self,state,action):
        #print("state is",state)
        chaser_state=state[0]
        opponent_state=state[1]
        if chaser_state==opponent_state and action=='tag':
            return (chaser_state,29) # chaser wins
        if action=='tag':
            return (chaser_state,self.run_away(state)) #chaser fucced up
        return (self.chase(state[0],action),self.run_away(state))

    def x_coord(self,state):
        if state<20:
            return state%10
        return (state-20)%3+5
    def y_coord(self,state):
        if state<20:
            return state//10
        return state//10+(state-20)//3

    def run_away(self,state):
        if random.random()<0.2:
            return state[1]
        actions=[]
        if self.x_coord(state[0])<=self.x_coord(state[1]):
            actions.append('e')
        if self.x_coord(state[0])>=self.x_coord(state[1]):
            actions.append('w')
        if self.y_coord(state[0])<=self.y_coord(state[1]):
            actions.append('n')
        if self.y_coord(state[0])>=self.y_coord(state[1]):
            actions.append('s')

        good_actions=[]
        for a in actions:
            if not self.out_of_bounds(state[1],a):
                good_actions.append(a)
        if not good_actions:
            actions=['e','w','n','s']
            for a in actions:
                if not self.out_of_bounds(state[1],a):
                    good_actions.append(a)

        action=random.choice(good_actions)
        return self.chase(state[1],action)

    def out_of_bounds(self,state,action):
        if state==0 and action=='w':
            return True
        if state==9 and action=='e':
            return state
        if state<10 and action=='s':
            return True
        if state<20 and state>=10:
            if state<15 and state>17 and action=='n':
                return True
            if state==10 and action=='w':
                return True
            if state==19 and action=='e':
                return True
        if state==20 and action=='w':
            return True
        if state==22 and action=='e':
            return True
        if state==23 and action=='w':
            return True
        if state==25 and action=='e':
            return True
        if state>=26 and action=='n':
            return True
        if state==26 and action=='w':
            return True
        if state==28 and action=='e':
            return True
        return False

    def chase(self,state,action):
        if self.out_of_bounds(state,action):
            return state
        if action=='e':
            return state+1
        if action=='w':
            return state-1
        if state<10 and action=='n':
            if action=='n':
                return state+10
        if state<20:
            if action=='n':
                return state+5
            if action=='s':
                return state-10
        if state<23:
            if action=='s':
                return state-5
            if action=='n':
                return state+3
        if state<26:
            if action=='s':
                return state-3
            if action=='n':
                return state+3
        if action=='s':
            return state-3


    def stepInTime(self):
        self.t+=1


    def getObs(self,state):
        if state[0]==state[1]:
            return True
        return False

    def get_g(self,alpha,action):
        if action=='n':
            a=0
        if action=='s':
            a=1
        if action=='e':
            a=2
        if action=='w':
            a=3
        if action=='tag':
            a=4
        ptrans=np.squeeze(self.pt[a,:,:,:,:])
        g1=np.einsum('ijkl,kl,kl->ij',ptrans,self.p_obs[0],alpha)
        g2=np.einsum('ijkl,kl,kl->ij',ptrans,self.p_obs[1],alpha)
        return [g1,g2]

    def getReward(self,state,action):
        if action=='tag':
            if state[0]==state[1]:
                return 10
            return -10
        return -1

    def reset(self,r=None):
        if r is None:
            self.pos=random.randrange(29)
        else:
            self.pos=r
        self.t=0

def unravel_belief(b):
    out=np.zeros_like(b)
    dims0=(out.shape[0]+1)//2
    dims1=(out.shape[1]+1)//2
    for i in range(-(dims0-1),dims0):
        for j in range(-(dims1-1),dims1):
            new_i=i+dims0-1
            new_j=j+dims1-1
            out[new_i,new_j]=b[i,j]
    return out
