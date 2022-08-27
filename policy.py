import random
import numpy as np
from datetime import datetime
from scipy.stats import entropy

class GreedyDiscrimination:
    def __init__(self,ag,r=2,eps=0):
        self.ag=ag
        self.r=r
        self.eps=eps
    def getAction(self,belief):
        if random.random()<self.eps:
            actions=self.ag.env.actions.copy()
            actions.remove("abort")
            return random.choice(actions)

        if np.sum(belief[1,:,:])>self.r*np.sum(belief[0,:,:]):
            return("abort")
        b=np.squeeze(belief[0,:,:])
        belief_pos = np.unravel_index(np.argmax(b),b.shape)
        action=random.choice(follow_loc(belief_pos,self.ag.true_pos))
        return self.ag.env.array_to_str(action)

class InfotacticDiscrimination:
    def __init__(self,ag,r=2,eps=0):
        self.agent=ag
        self.r=r
        self.eps=eps
    def getAction(self,belief):
        if random.random()<self.eps:
            actions=self.ag.env.actions.copy()
            actions.remove("abort")
            return random.choice(actions)
        if np.sum(belief[1,:,:])>self.r*np.sum(belief[0,:,:]):
            return("abort")
        deltaS=np.inf
        S=entropy(belief)
        #print("entropy: ",S)
        best_action=None
        for action in self.agent.env.actions:
            if action=="abort":
                continue
            #print(action)
            newpos=self.agent.belief_env.transition(self.agent.true_pos,action)
            p=np.sum(belief[:,newpos[0],newpos[1]])
            #print("probability of finding source: ",p)
            x=newpos[0]-np.arange(self.agent.belief_env.dims[0])
            y=newpos[1]-np.arange(self.agent.belief_env.dims[1])
            #rates=self.agent.belief_env.get_rate(x[:,None],y[None,:])
            # if self.poisson:
            #     h=np.sum(rates*belief)
            #     l_hit=1-np.exp(-h)
            # else:
            l=np.stack([self.agent.belief_env.get_likelihood(x[:,None],y[None,:],sep=0),self.agent.belief_env.get_likelihood(x[:,None],y[None,:],sep=1)],axis=0)
            l_hit=np.sum(l*belief)
            l_miss=1-l_hit-p
            #print("probability of hit: ",l)
            s0=entropy(self.agent.computeBelief(False,belief,action))
            #print("entropy associated with no hit:",s0)
            s1=entropy(self.agent.computeBelief(True,belief,action))
            #print("entropy associated with hit:",s1)
            tmp=p*(-S)+(1-p)*(l_miss*(s0-S)+l_hit*(s1-S))
            #print("expected DeltaS: ",tmp)
            if tmp<deltaS:
                deltaS=tmp
                best_action=action
        #print("best action is: ",best_action)

        if np.array_equal(best_action,None):
            raise RuntimeError('Failed to find optimal action')
        return best_action


class RandomPolicy2D:
    def __init__(self):
        self.seed=datetime.now()
        random.seed(self.seed)

    def getAction(self,belief):
        n=random.randrange(1,5)
        if n==0:
            return np.array([0,0])
        elif n==1:
            return np.array([1,0])
        elif n==2:
            return np.array([-1,0])
        elif n==3:
            return np.array([0,1])
        else:
            return np.array([0,-1])

class CastAndSurge:
    def __init__(self,agent):
        self.agent=agent
        self.tmax=0
        self.tau=0
        self.vertical_dir=1
    def getAction(self,belief):
        if self.agent.last_obs:
            self.tmax=0
            self.tau=0
            return np.array([-1,0])
        if self.tau<self.tmax:
            if self.vertical_dir==1:
                self.tau+=1
                return np.array([0,1])
            else:
                self.tau+=1
                return np.array([0,-1])
        if self.tau==self.tmax:
            self.tmax+=1
            self.tau=0
            self.vertical_dir=self.vertical_dir*(-1)
            return np.array([-1,0])


class RandomBanditPolicy:
    def __init__(self,k):
        self.seed=datetime.now()
        random.seed(self.seed)
        self.k=k

    def getAction(self,belief):
        return random.randrange(self.k)

class RandomPolicy:
    def __init__(self,env):
        self.seed=datetime.now()
        random.seed(self.seed)
        self.actions=env.actions

    def getAction(self,belief):
        return random.choice(self.actions)

class ThompsonSampling:
    def __init__(self,agent,persistence_time=1):
        self.agent=agent
        self.persistence_time=persistence_time
        self.t=0
        self.location=None
    def reset(self):
        self.t=0
        self.location=None
    def getAction(self,belief):
        #print('heading towards',self.location)
        if np.array_equal(self.location,self.agent.true_pos):
            self.t=0
        if self.t==0:
            b=belief.flatten()
            indices=np.arange(b.shape[0])
            index=np.random.choice(indices,p=b)
            self.location=np.unravel_index(index,belief.shape)
        self.t+=1
        if self.t==self.persistence_time:
            self.t=0
        return random.choice(follow_loc(self.location,self.agent.true_pos))

class QMDPPolicy:
    def __init__(self,agent,gamma=None):
        self.agent=agent
        if gamma is not None:
            self.q=gamma**(self.agent.env.dist)
            self.q[0,0]=0
        else:
            self.q=self.agent.env.qvalue

    def getAction(self,belief):
        b=self.agent.perseus_belief(belief)
        bestaction=None
        best_r=-np.inf
        for a in self.agent.env.actions:
            q=np.roll(self.q,tuple(-a),axis=(0,1))
            if np.array_equal(a,[1,0]):
                q[self.agent.env.dims[0],:]=0
            if np.array_equal(a,[-1,0]):
                q[self.agent.env.dims[0]-1,:]=0
            if np.array_equal(a,[0,1]):
                q[:,self.agent.env.dims[1]-1]=0
            if np.array_equal(a,[0,-1]):
                q[:,self.agent.env.dims[1]]=0
            r=np.sum(b*q)
            #print(a,r)
            if r>best_r:
                best_r=r
                bestaction=a
        return bestaction

class TrivialPolicy:
    def __init__(self):
        pass
    def getAction(self,belief):
        return np.array([0,0])

class ActionVoting:
    def __init__(self,agent):
        self.agent=agent
        dims=self.agent.env.dims

        self.delta_mat=np.zeros((self.agent.env.numactions,2*dims[0]-1,2*dims[1]-1))
        for i in range(-(dims[0]-1),dims[0]):
            for j in range(-(dims[1]-1),dims[1]):
                actions=follow_loc([0,0],[i,j])
                for action in range(self.agent.env.numactions):
                    if any((self.agent.env.actions[action] == x).all() for x in actions):
                        self.delta_mat[action,i,j]=1

    def getAction(self,belief):
        b=self.agent.perseus_belief(belief)
        bestaction=None
        best_p=-np.inf
        for a in range(self.agent.env.numactions):
            p=np.sum(self.delta_mat[a,:,:]*b)
            if p>best_p:
                best_p=p
                bestaction=a
        return self.agent.env.actions[bestaction]

class OptimalPolicy:
    def __init__(self,vf,agent,parallel=False,epsilon=0):
        self.vf=vf
        self.agent=agent
        self.parallel=parallel
        self.set_used=False
        self.epsilon=epsilon
        self.last_value=None
    def getAction(self,belief):
        if random.random()<self.epsilon:
            return random.choice(self.agent.env.actions)
        b=self.agent.perseus_belief(belief) # this line for situations where belief used by perseus is defined differently
        value,bestalpha=self.vf.value(b,parallel=self.parallel)
        if self.set_used:
            bestalpha.used=True
        self.last_value=value
        return bestalpha.action

class RandomPolicyTiger:
    def __init__(self):
        self.seed=datetime.now()
        random.seed(self.seed)

    def getAction(self,belief):
        return random.randrange(3)

class GreedyPolicy:
    def __init__(self,agent):
        self.agent=agent

    def getAction(self,belief):
        location = np.unravel_index(np.argmax(belief),belief.shape)
        return random.choice(follow_loc(location,self.agent.true_pos))

class InfotacticPolicyOriginal:
    def __init__(self,agent):
        self.agent=agent

    def getAction(self,belief):
        deltaS=np.inf
        S=entropy(belief)
        print("entropy: ",S)
        best_action=None
        for action in self.agent.env.actions:
            print(action)
            #newpos=self.agent.belief_env.transition(self.agent.true_pos,action)


            transported_belief=self.agent.transportBelief(belief,action)
            p=transported_belief[self.agent.env.x0,self.agent.env.y0]
            print("probability of finding source: ",p)
            x=np.arange(self.agent.env.dims[0])-self.agent.env.x0
            y=np.arange(self.agent.env.dims[1])-self.agent.env.y0
            rates=self.agent.belief_env.get_rate(x[:,None],y[None,:])
            h=np.sum(rates*transported_belief)
            l=1-np.exp(-h)
            print("probability of hit: ",l)
            pprime=(1-self.agent.env.likelihood)*transported_belief
            pprime=pprime/np.sum(pprime)
            s0=entropy(pprime)
            print("entropy associated with no hit:",s0)
            pprime=self.agent.env.likelihood*transported_belief
            pprime=pprime/np.sum(pprime)
            s1=entropy(pprime)
            print("entropy associated with hit:",s1)
            tmp=p*(-S)+(1-p)*((1-l)*(s0-S)+l*(s1-S))
            print("expected DeltaS: ",tmp)
            if tmp<deltaS:
                deltaS=tmp
                best_action=action
        print("best action is: ",best_action)

        if np.array_equal(best_action,None):
            raise RuntimeError('Failed to find optimal action')
        return best_action

class SpaceAwareInfotaxis:
    def __init__(self,agent,epsilon=0,type=0):
        self.agent=agent
        self.epsilon=epsilon
        self.type=type

    def getAction(self,belief):
        if np.random.random()<self.epsilon:
            return np.random.choice(self.ag.env.actions)
        newJ=np.inf
        #S=np.log2(2**(entropy(belief,base=2)-1)-0.5+abs
        #print("entropy: ",S)
        best_action=None

        for action in self.agent.env.actions:
            #print(action)
            newpos=self.agent.belief_env.transition(self.agent.true_pos,action)
            p=belief[newpos[0],newpos[1]]
            #print("probability of finding source: ",p)
            x=newpos[0]-np.arange(self.agent.belief_env.dims[0])
            y=newpos[1]-np.arange(self.agent.belief_env.dims[1])
            #rates=self.agent.belief_env.get_rate(x[:,None],y[None,:])
            # if self.poisson:
            #     h=np.sum(rates*belief)
            #     l_hit=1-np.exp(-h)
            # else:
            l_hit=np.sum(self.agent.belief_env.get_likelihood(x[:,None],y[None,:])*belief)
            l_miss=1-l_hit-p
            #print("probability of hit: ",l)
            x=np.arange(self.agent.belief_env.dims[0])
            y=np.arange(self.agent.belief_env.dims[1])
            dist=np.abs(newpos[0]-x[:,None])+np.abs(newpos[1]-y[None,:])
            self.dist=np.zeros(self.agent.belief_env.dims)
            b0=self.agent.computeBelief(False,belief,action)
            b1=self.agent.computeBelief(True,belief,action)
            if self.type==0:
                J0=np.log2(max([2**(entropy(b0.flatten(),base=2)-1)-0.5+np.sum(dist*b0),1]))
                J1=np.log2(max([2**(entropy(b1.flatten(),base=2)-1)-0.5+np.sum(dist*b1),1]))
            elif self.type==1:
                J0=np.log2(2**(entropy(b0.flatten(),base=2)-1)+0.5+np.sum(dist*b0))
                J1=np.log2(2**(entropy(b1.flatten(),base=2)-1)+0.5+np.sum(dist*b1))
            else:
                raise RuntimeError("invalid policy subtype")
            #print("entropy associated with hit:",s1)
            tmp=l_miss*J0+l_hit*J1
            #print("expected DeltaS: ",tmp)
            if tmp<newJ:
                newJ=tmp
                best_action=action
        #print("best action is: ",best_action)

        if np.array_equal(best_action,None):
            raise RuntimeError('Failed to find optimal action')
        return best_action

class InfotacticPolicy:
    def __init__(self,agent,poisson=True):
        self.agent=agent
        self.poisson=poisson
    def getAction(self,belief):
        newS=np.inf
        #S=entropy(belief)
        #print("entropy: ",S)
        best_action=None
        for action in self.agent.env.actions:
            #print(action)
            newpos=self.agent.belief_env.transition(self.agent.true_pos,action)
            p=belief[newpos[0],newpos[1]]
            #print("probability of finding source: ",p)
            x=newpos[0]-np.arange(self.agent.belief_env.dims[0])
            y=newpos[1]-np.arange(self.agent.belief_env.dims[1])
            #rates=self.agent.belief_env.get_rate(x[:,None],y[None,:])
            # if self.poisson:
            #     h=np.sum(rates*belief)
            #     l_hit=1-np.exp(-h)
            # else:
            l_hit=np.sum(self.agent.belief_env.get_likelihood(x[:,None],y[None,:])*belief)
            l_miss=1-l_hit-p
            #print("probability of hit: ",l)
            s0=entropy(self.agent.computeBelief(False,belief,action).flatten())
            #print("entropy associated with no hit:",s0)
            s1=entropy(self.agent.computeBelief(True,belief,action).flatten())
            #print("entropy associated with hit:",s1)
            tmp=l_miss*s0+l_hit*s1
            #print("expected DeltaS: ",tmp)
            if tmp<newS:
                newS=tmp
                best_action=action
        #print("best action is: ",best_action)

        if np.array_equal(best_action,None):
            raise RuntimeError('Failed to find optimal action')
        return best_action

# def entropy(belief,base=None):
#     return np.sum(entr(belief,base=base))

def follow_loc(location,true_pos):
    if location[1]==true_pos[1]:
        if location[0]>true_pos[0]:
            return [np.array([1,0])]
        return [np.array([-1,0])]
    if location[0]==true_pos[0]:
        if location[1]>true_pos[1]:
            return [np.array([0,1])]
        return [np.array([0,-1])]
    x=[]
    if location[0]>true_pos[0]:
        x.append(np.array([1,0]))
    else:
        x.append(np.array([-1,0]))
    if location[1]>true_pos[1]:
        x.append(np.array([0,1]))
    else:
        x.append(np.array([0,-1]))
    return x
