import numpy as np
import environment
import policy
import sys
from scipy.special import gamma
import random
from timeit import default_timer as timer

class Agent:
    def __init__(self,e,r0,p=None):
        self.env=e
        self.policy=p
        self.true_pos=r0
        self.dims=e.dims
        self.belief=np.ones(self.dims)
        self.belief=self.belief/np.sum(self.belief)
        self.last_action=None
        self.rewards=[]
        self.last_reward=None
        self.last_obs=None


    def set_policy(self,p):
        policy=p

    def reset(self,r0,b=None):
        self.set_belief(b)
        self.set_position(r0)
        self.last_action=None
        self.rewards=[]
        self.last_reward=None
        self.last_obs=None

    def updateBelief(self,obs):
        pass
        #needs to be implemented on a case-by-case basis

    def stepInTime(self,values=False,make_obs=True,force_obs=None):
        #t1=timer()
        if make_obs:
            if force_obs is None:
                obs=self.env.getObs(self.true_pos)
            else:
                obs=force_obs

        #t2=timer()
        #print("obs time:",t2-t1)
        #t1=timer()
            self.updateBelief(obs)
            self.last_obs=obs
        #t2=timer()
        #print("belief update time:",t2-t1)
        b=self.perseus_belief(self.belief)
        #t1=timer()
        action=self.policy.getAction(self.belief)
        #t2=timer()
        #print("action selection time:",t2-t1)
        if values:
            v=self.policy.last_value
        #print(action,v)
        self.last_reward=self.env.getReward(self.true_pos,action)
        self.rewards.append(self.last_reward)
        self.true_pos=self.env.transition(self.true_pos,action)
        #obs=self.env.getObs(self.true_pos)

        self.last_action=action
        #self.last_obs=obs
        #self.belief=self.transportBelief(belief,action)
        if values:
            return b,v
        return b

    def set_position(self,pos):
        self.true_pos=pos

    def set_belief(self,b=None):
        if b==None:
            self.belief=np.ones(self.dims)
        else:
            self.belief=b
        self.belief=self.belief/np.sum(self.belief)

    def set_policy(self,p):
        self.policy=p

class DiscriminationAgent(Agent):
    def __init__(self,e,r0,p=None,belief_env=None):
        self.env=e
        self.policy=p
        self.true_pos=r0
        self.dims=(2,)+e.dims
        self.belief=np.ones(self.dims)
        self.belief=self.belief/np.sum(self.belief)
        self.last_action=None
        self.rewards=[]
        self.last_reward=None
        self.last_obs=None
        self.nhits=0
        self.ob=False
        if belief_env is not None:
            self.belief_env=belief_env #if the agent's model doesn't match the true model
        else:
            self.belief_env = e
    def stepInTime(self):
        b=self.perseus_belief(self.belief)
        action=self.policy.getAction(self.belief)
        # if self.env.outOfBounds(action+self.true_pos):
        #     self.ob=True
        # else:
        #     self.ob=False
        self.belief=self.transportBelief(self.belief,action)
        self.last_reward=self.env.getReward(self.true_pos,action)
        self.rewards.append(self.last_reward)
        self.true_pos=self.env.transition(self.true_pos,action)
        obs=self.env.getObs(self.true_pos)
        self.updateBelief(obs)
        self.last_action=action
        self.last_obs=obs
        return b

    def reset(self,r0,b=None):
        super().reset(r0,b)
        self.nhits=0

    def updateBelief(self,obs):
        if obs==True:
            self.nhits+=1
        self.belief=self.computeBelief(obs,self.belief)

    def computeBelief(self,obs,belief,action="null"):
        if obs=="source":
            b=np.zeros(self.dims)
            b[0,0,0]=1
            return b
        out=belief.copy()
        pos=self.true_pos+self.env.str_to_array(action)
        x=np.arange(pos[0],pos[0]-self.dims[1],-1)
        y=np.arange(pos[1],pos[1]-self.dims[2],-1)
        l0=self.env.get_likelihood(x[:,None],y[None,:],sep=0)
        l1=self.env.get_likelihood(x[:,None],y[None,:],sep=1)
        l=np.stack([l0,l1],axis=0)
        if obs==True:
            out=out*l
        else:
            out=out*(1-l)
        out[:,0,0]=0
        if np.sum(out)==0:
            raise RuntimeError('zero belief encountered at pos '+str(pos)+', time '+str(self.env.t))
        out=out/np.sum(out)
        return out

    def transportBelief(self,belief,action):
        return belief

    def perseus_belief(self,belief):
        dims=belief.shape
        b=np.pad(belief,((0,0),(0,dims[1]-1),(0,dims[2]-1)))
        b=np.flip(b)
        b=np.roll(b,(1+self.true_pos[0],1+self.true_pos[1]),axis=(1,2))
        return b



class MosquitoAgentOriginal(Agent):
    def __init__(self,e,r0,p=None,belief_env=None):
        self.env=e
        self.policy=p
        self.true_pos=r0
        self.dims=e.dims
        self.belief=np.ones(self.dims)
        self.belief=self.belief/np.sum(self.belief)
        self.last_action=None
        self.rewards=[]
        self.last_reward=None
        self.last_obs=None
        self.nhits=0
        self.ob=False
        if belief_env is not None:
            self.belief_env=belief_env #if the agent's model doesn't match the true model
        else:
            self.belief_env = e

    def stepInTime(self):
        b=self.perseus_belief(self.belief)
        action=self.policy.getAction(self.belief)
        if self.env.outOfBounds(action+self.true_pos):
            self.ob=True
        else:
            self.ob=False
        self.belief=self.transportBelief(self.belief,action)

        self.last_reward=self.env.getReward(self.true_pos,action)
        self.rewards.append(self.last_reward)
        self.true_pos=self.env.transition(self.true_pos,action)
        obs=self.env.getObs(self.true_pos)
        self.updateBelief(obs)
        self.last_action=action
        self.last_obs=obs

        return b

    def reset(self,r0,b=None):
        super().reset(r0,b)
        self.nhits=0

    def updateBelief(self,obs):
        if obs==True:
            self.nhits+=1
        self.belief=self.computeBelief(obs,self.belief)

    def computeBelief(self,obs,belief):
        #transport belief
        # if action is None:
        #     action=np.array([0,0])
        # out=self.transportBelief(belief,action)
        #bayesian update
        out=belief.copy()
        l=self.belief_env.likelihood
        if obs==True:
            out=out*l
        else:
            out=out*(1-l)
        out[self.env.x0,self.env.y0]=0
        if np.sum(out)==0:
            raise RuntimeError('zero belief encountered at pos '+str(pos)+', time '+str(self.env.t))
        out=out/np.sum(out)
        return out

    def transportBelief(self,belief,action):
        out=belief.copy()
        if self.ob==True:
            return out #belief not transported if agent tried to leave the simulation box

        if (action==np.array([1,0])).all():
            tmp=np.array(out[-1,:])
            out[1:,:]=out[:-1,:]
            out[0,:]=0
            out[-1,:]+=tmp
        elif (action==np.array([-1,0])).all():
            tmp=np.array(out[0,:])
            out[:-1,:]=out[1:,:]
            out[-1,:]=0
            out[0,:]+=tmp
        elif (action==np.array([0,1])).all():
            tmp=np.array(out[:,-1])
            out[:,1:]=out[:,:-1]
            out[:,0]=0
            out[:,-1]+=tmp
        elif (action==np.array([0,-1])).all():
            tmp=np.array(belief[:,0])
            out[:,:-1]=out[:,1:]
            out[:,-1]=0
            out[:,0]+=tmp
        return out/np.sum(out)
    def perseus_belief(self,belief):
        return belief

class MosquitoAgent(Agent):
    def __init__(self,e,r0,p=None,belief_env=None):
        super().__init__(e,r0,p)
        self.nhits=0
        self.ob=False
        self.stuck_count=0
        self.prev_pos=None
        self.prev_prev_pos=None
        if belief_env is not None:
            self.belief_env=belief_env #if the agent's model doesn't match the true model
        else:
            self.belief_env = e

    def stepInTime(self,values=False,make_obs=True,force_obs=None):
        if self.prev_pos is not None:
            self.prev_prev_pos=self.prev_pos.copy()
        self.prev_pos=self.true_pos.copy()
        b=super().stepInTime(values=values,make_obs=make_obs,force_obs=force_obs)
        if self.prev_prev_pos is not None and np.array_equal(self.prev_prev_pos,self.true_pos):
            self.stuck_count+=1
        else:
            self.stuck_count=0
        if self.env.outOfBounds(self.last_action+self.true_pos):
            self.ob=True
        else:
            self.ob=False
        return b

    def reset(self,r0,b=None):
        super().reset(r0,b)
        self.nhits=0

    def updateBelief(self,obs):
        super().updateBelief(obs)
        #if obs>=1:
        self.nhits+=obs
        self.belief=self.computeBelief(obs,self.belief)

    def computeBelief(self,obs,belief,action=np.array([0,0])):
        #transport belief
        #out=self.transportBelief(belief,action)

        out=belief.copy()
        #bayesian update


        pos=self.belief_env.transition(self.true_pos,action)
        if np.array_equal(pos,np.array([self.env.x0,self.env.y0])):
            out=out*0
            out[pos[0],pos[1]]=1
            return out
        #l=self.env.likelihood[self.true_pos[0]:self.true_pos[0]-self.dims[0],self.true_pos[1]:self.true_pos[1]-self.dims[1]]
        x=np.arange(pos[0],pos[0]-self.dims[0],-1)
        y=np.arange(pos[1],pos[1]-self.dims[1],-1)
        l=self.belief_env.get_likelihood(x[:,None],y[None,:],obs)
        out=out*l

        if np.sum(out)==0:
            raise RuntimeError('zero belief encountered at pos '+str(pos)+', time '+str(self.env.t))
        out=out/np.sum(out)
        return out

    #this function deprecated; was used for an older version where belief encoded location of mosquito, rather than source
    def transportBelief(self,belief,action):
        out=belief.copy()
        if self.ob==True:
            return out #no belief transported if agent tried to leave the simulation box

        if (action==np.array([1,0])).all():
            tmp=np.array(out[-1,:])
            out[1:,:]=out[:-1,:]
            out[0,:]=0
            out[-1,:]+=tmp
        elif (action==np.array([-1,0])).all():
            tmp=np.array(out[0,:])
            out[:-1,:]=out[1:,:]
            out[-1,:]=0
            out[0,:]+=tmp
        elif (action==np.array([0,1])).all():
            tmp=np.array(out[:,-1])
            out[:,1:]=out[:,:-1]
            out[:,0]=0
            out[:,-1]+=tmp
        elif (action==np.array([0,-1])).all():
            tmp=np.array(belief[:,0])
            out[:,:-1]=out[:,1:]
            out[:,-1]=0
            out[:,0]+=tmp
        return out
    def perseus_belief(self,belief):
        dims=belief.shape
        b=np.pad(belief,((0,dims[0]-1),(0,dims[1]-1)))
        b=np.flip(b)
        b=np.roll(b,(1+self.true_pos[0],1+self.true_pos[1]),axis=(0,1))
        return b

class CorrAgent(MosquitoAgent):
    def updateBelief(self,obs,action):
        self.nhits+=obs
        self.belief=self.computeBelief(obs,action,self.belief)

    def computeBelief(self,obs,last_action,b,action=np.array([0,0])):

        out=b.copy()
        #bayesian update

        pos=self.belief_env.transition(self.true_pos,action)
        x=np.arange(pos[0],pos[0]-self.dims[0],-1)
        y=np.arange(pos[1],pos[1]-self.dims[1],-1)
        l=self.belief_env.get_likelihood(x[:,None],y[None,:],obs,self.last_obs,last_action)
        out=out*l
        out[pos[0],pos[1]]=0
        if np.sum(out)==0:
            raise RuntimeError('zero belief encountered at pos '+str(pos)+', time '+str(self.env.t))
        out=out/np.sum(out)
        return out

    def stepInTime(self,values=False,make_obs=True,force_obs=None,obs_aware_policy=False,corr_aware=True):
        if self.prev_pos is not None:
            self.prev_prev_pos=self.prev_pos.copy()
        self.prev_pos=self.true_pos.copy()
        if make_obs:
            if force_obs is None:
                obs=self.env.getObs(self.true_pos,ag=self)
            else:
                obs=force_obs
            #self.last_obs=obs
          #  print("updating belief with last_obs=",obs,"and last_action=",self.last_action)
            if corr_aware:    
                self.updateBelief(obs,self.last_action)
            else:
                self.updateBelief(obs,None)
            self.last_obs=obs

        b=self.perseus_belief(self.belief)
        if obs_aware_policy:
            action=self.policy.getAction(self.belief,self.last_obs)
        else:
            action=self.policy.getAction(self.belief)

        if values:
            v=self.policy.last_value

        self.last_reward=self.env.getReward(self.true_pos,action)
        self.rewards.append(self.last_reward)
        self.true_pos=self.env.transition(self.true_pos,action)

        self.last_action=action
        if self.prev_prev_pos is not None and np.array_equal(self.prev_prev_pos,self.true_pos):
            self.stuck_count+=1
        else:
            self.stuck_count=0
        if self.env.outOfBounds(self.last_action+self.true_pos):
            self.ob=True
        else:
            self.ob=False
        if values:
            return b,v





class TigerAgent(Agent):
    def updateBelief(self,obs):
        super().updateBelief(obs)
        if obs==4:
            self.belief=np.zeros((33,))
            self.belief[-1]=1
            return
        tmp=np.zeros((33,))
        #transport belief
        for i in range(32):
            new_pos=self.env.transition(i,self.last_action)
            tmp[new_pos]+=self.belief[i]
        tmp[:-1]+=self.belief[-1]/32 #random propagation from goal to new state
        assert np.abs(np.sum(tmp)-1)<0.01
        tmp=tmp/np.sum(tmp)

        self.belief=tmp

        self.belief[32]=0
        if obs==0: #nothing
            for i in range(32):
                if self.env.facingWall(i):
                    self.belief[i]=0
                if i<4 or (12<=i and i<16):
                    self.belief[i]=0
        if obs==1: #wall
            for i in range(32):
                if not self.env.facingWall(i):
                    self.belief[i]=0
                if i<4 or (12<=i and i<16):
                    self.belief[i]=0
        if obs==2: #tiger
            for i in range(32):
                if self.env.facingWall(i):
                    self.belief[i]=0
                if not (i<4 or (12<=i and i<16)):
                    self.belief[i]=0
        if obs==3: # tiger + wall
            for i in range(32):
                if not self.env.facingWall(i):
                    self.belief[i]=0
                if not (i<4 or (12<=i and i<16)):
                    self.belief[i]=0
        self.belief=self.belief/np.sum(self.belief)

class BanditAgent:
    def __init__(self,e,p=None):
        self.env=e
        self.policy=p
        self.alphas=np.ones((self.env.numactions,))
        self.betas=np.ones((self.env.numactions,))
        self.p_grid=e.p_grid
        self.belief=self.get_belief(self.p_grid)
        self.reward=0
        self.rewards=[0]
        self.last_action=None
        self.last_obs=None

    def perseus_belief(self,belief):
        y=np.outer(belief[0,:],belief[1,:])
        Np=len(self.p_grid)
        dims=(Np,Np)
        for a in range(2,self.env.numactions):
            dims=dims+(Np,)
            y=np.reshape(np.outer(y,belief[a,:]),dims)
        return y

    def set_policy(self,pol):
        self.policy=pol

    def reset(self):
        self.reward=0
        self.rewards=[0]
        self.alphas=np.ones((self.env.numactions,))
        self.betas=np.ones((self.env.numactions,))
        self.belief=self.get_belief(self.p_grid)
        self.last_action=None
        self.last_obs=None

    def get_belief(self,x):
        alpha=self.alphas[:,None]
        beta=self.betas[:,None]
        x=x[None,:]
        y=x**(alpha-1)*(1-x)**(beta-1)
        for a in range(0,self.env.numactions):
            y[a,:]=y[a,:]/np.sum(y[a,:])
        return y

    def stepInTime(self):
        action=self.policy.getAction(self.belief)
        obs=self.env.getObs(action)
        if obs:
            self.reward+=self.env.gamma**self.env.t
        self.last_obs=obs
        self.rewards.append(self.reward)
        self.updateBelief(obs,action)
        self.env.stepInTime()
        self.last_action=action

    def updateBelief(self,obs,action):
        if obs:
            self.alphas[action]+=1
        if not obs:
            self.betas[action]+=1
        self.belief=self.get_belief(self.p_grid)

    def set_belief(self,alphas,betas):
        self.alphas=alphas
        self.betas=betas
        self.belief=self.get_belief(self.p_grid)

class TagAgent:
    def __init__(self,e,p=None):
        self.env=e
        self.policy=p
        self.true_pos=random.randrange(29)
        self.dims=e.dims
        self.belief=np.zeros(self.dims)
        self.belief[self.true_pos,:]=1
        self.belief[:,29]=0
        #self.belief[:,self.true_pos]=0
        self.belief=self.belief/np.sum(self.belief)
        self.last_action=None
        self.last_reward=None

    def set_policy(self,p):
        policy=p

    def reset(self,r=None):
        if r is None:
            self.true_pos=random.randrange(29)
        else:
            self.true_pos=r

        self.belief=np.zeros(self.dims)
        self.belief[self.true_pos,:]=1
        #self.belief[:,self.true_pos]=0
        self.belief[:,29]=0
        self.belief=self.belief/np.sum(self.belief)
        self.last_action=None
        self.last_reward=None

    def updateBelief(self,obs,action):
        if action is None:
            if obs==False:
                self.belief=np.zeros(self.dims)
                self.belief[self.true_pos,:]=1
                self.belief[:,29]=0
                self.belief[:,self.true_pos]=0
                self.belief=self.belief/np.sum(self.belief)
            if obs==True:
                self.belief=np.zeros(self.dims)
                self.belief[self.true_pos,self.true_pos]=1
            return
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
        if obs==True:
            po=self.env.p_obs[1]
        else:
            po=self.env.p_obs[0]
        x=np.einsum('kl,ijkl,ij->kl',po,np.squeeze(self.env.pt[a,:,:,:,:]),self.belief)
        if np.sum(x)==0:
            raise RuntimeError('zero belief encountered after observation '+str(obs))
        self.belief=x/np.sum(x)


    def stepInTime(self):
        print("true pos: ",self.true_pos)
        print("chasee pos: ",self.env.pos)
        obs=self.env.getObs((self.true_pos,self.env.pos))
        self.updateBelief(obs,self.last_action)
        action=self.policy.getAction(self.belief)
        self.last_reward=self.env.getReward((self.true_pos,self.env.pos),action)
        s=self.env.transition((self.true_pos,self.env.pos),action)
        self.true_pos=s[0]
        self.env.pos=s[1]
        self.env.stepInTime()
        self.last_action=action


    def set_position(self,pos):
        self.true_pos=pos

    def set_belief(self,b=None):
        if b==None:
            self.belief=np.ones(self.dims)
        else:
            self.belief=b
        self.belief=self.belief/np.sum(self.belief)

    def set_policy(self,p):
        self.policy=p

    def perseus_belief(self,b):
        return b
