import numpy as np
import environment
import policy
import agent
import sys
import perseus_redux as perseus_redux
import random
from datetime import datetime
import pickle
from copy import copy
import os
from timeit import default_timer as timer

def policy_trials(ag,env,starts,n_trials,thompson=False,tmax=1000,x0=10,y0=20,errors=False,vf=None,wait=None,no_nn=False,error_frac=0.3333):
    n_starts=len(starts)
    times=np.zeros((n_starts,n_trials))
    rewards=np.zeros((n_starts,n_trials))
    normalized_times=np.zeros((n_starts,n_trials))
    failure=0
    start=timer()
    total_time=0
    error_time=0
    if errors:
        bellman=[]
    else:
        bellman=None
    for j in range(n_starts):
        #if errors:
            #new_bellman=[]
        for l in range(n_trials):
            ag.reset(starts[j])
            if thompson:
                ag.policy.reset()
            pol=copy(ag.policy)
            ag.set_policy(policy.TrivialPolicy())
            if wait is not None:
                while True:
                    ag.reset(starts[j])
                    for t in range(wait):
                        ag.stepInTime()
                        env.stepInTime()
                        if ag.last_obs:
                            break
                    if ag.last_obs:
                        break
            ag.set_policy(pol)
            if not ag.last_obs==True:
                ag.updateBelief(True)
            if no_nn:
                ag.belief[ag.true_pos[0]-1,ag.true_pos[1]]=0
                ag.belief[ag.true_pos[0]+1,ag.true_pos[1]]=0
                ag.belief[ag.true_pos[0],ag.true_pos[1]-1]=0
                ag.belief[ag.true_pos[0],ag.true_pos[1]+1]=0
                ag.belief=ag.belief/np.sum(ag.belief)
            rew=0
            for k in range(tmax):
                #print('agent at',ag.true_pos)
                t1=timer()
                if errors:
                    b,v=ag.stepInTime(values=True)
                else:
                    b=ag.stepInTime()
                t2=timer()
                total_time+=t2-t1
                if errors and random.random()<error_frac:
                    t1=timer()
                    bellman.append(vf.bellmanErrorNew(b,value=v))
                    t2=timer()
                    error_time+=t2-t1
                env.stepInTime()
                #if ag.last_reward is None:
                    #print(ag.true_pos)
                    #print(ag.last_action)
                try:
                    rew+=env.gamma**k*ag.last_reward
                except:
                    print(env.gamma)
                    print(k)
                    print(ag.last_reward)
                    print(ag.true_pos)
                    print(ag.last_action)
                    raise RuntimeError('reward updating failed')    
                if ag.true_pos[0]==env.x0 and ag.true_pos[1]==env.y0:
                    #print('found source at t=',k+1)
                    break
            rewards[j,l]=rew
            times[j,l]=k+1
            if k+1==tmax:
            	failure+=1
            d=np.abs(starts[j][0]-x0)+np.abs(starts[j][1]-y0)
            normalized_times[j,l]=(k+1)/d-1
        #if errors:
            #bellman.append(new_bellman)
    end=timer()
    failure=failure/(n_starts*n_trials)
    reward_averages=np.mean(rewards,axis=0)
    time_averages=np.mean(times,axis=0)
    time_av=np.mean(time_averages)
    reward_av=np.mean(reward_averages)
    time_std=np.std(time_averages)
    reward_std=np.std(reward_averages)
    normalized_averages=np.mean(normalized_times,axis=0)
    norm_av=np.mean(normalized_averages)
    norm_std=np.std(normalized_averages)
    return rewards,times,time_av,reward_av,time_std,reward_std,norm_av,norm_std,failure,bellman,end-start,total_time,error_time

def generate_beliefs(ag,env,tmax,n_collect,wait=None,x0=None,y0=None,beliefs=None,distances=None,belief_dict=None,close=False,hit_at_first=False,ordered=False,random_walk=0,no_nearest_neighbors=False,test_min=0.0015,test_max=0.004,min_error=None,vf=None):
    if beliefs is None:
        new_beliefs=[]
    else:
        new_beliefs=beliefs
    if distances is None and ordered:
        distances=[]
    if belief_dict is None:
        new_belief_dict={}
    else:
        new_belief_dict=belief_dict
    n_b=0
    rand_x0=False
    rand_y0=False
    if x0 is None:
        rand_x0=True
    if y0 is None:
        rand_y0=True
    while n_b<n_collect:
        if rand_x0:
            x0=random.randrange(int(env.dims[0]/2))
        if rand_y0:
            y0=random.randrange(int(env.dims[1]/3),2*int(env.dims[1]/3))
        if not close:
            xstart=random.randrange(env.dims[0])
            ystart=random.randrange(env.dims[1])
            while not in_cone(env,[xstart,ystart],[x0,y0],tol=test_min) or in_cone(env,[xstart,ystart],[x0,y0],tol=test_max):
                xstart=random.randrange(env.dims[0])
                ystart=random.randrange(env.dims[1])
        else:
            xstart=random.randrange(x0-3,x0+4)
            ystart=random.randrange(y0-3,y0+4)
            while xstart==x0 and ystart==y0:
                xstart=random.randrange(x0-3,x0+4)
                ystart=random.randrange(y0-3,y0+4)
        print([xstart,ystart])
        ag.reset([xstart,ystart])
        env.reset()
        env.set_pos(x0,y0)
        pol=copy(ag.policy)
        random_pol=policy.RandomPolicy2D()
        ag.set_policy(random_pol)
        for i in range(random_walk):
            ag.stepInTime()
            env.stepInTime()
            if ag.last_obs==True:
                break
        if wait is not None:
            trivial=policy.TrivialPolicy()
            ag.set_policy(trivial)
            while True:
                ag.reset([xstart,ystart])
                for i in range(wait):
                    ag.stepInTime()
                    env.stepInTime()
                    if ag.last_obs:
                        break
                if ag.last_obs:
                    break


        ag.set_policy(pol)

        if hit_at_first:
            if not ag.last_obs==True:
                ag.updateBelief(True)
        if no_nearest_neighbors:
            ag.belief[ag.true_pos[0]-1,ag.true_pos[1]]=0
            ag.belief[ag.true_pos[0]+1,ag.true_pos[1]]=0
            ag.belief[ag.true_pos[0],ag.true_pos[1]-1]=0
            ag.belief[ag.true_pos[0],ag.true_pos[1]+1]=0
            ag.belief=ag.belief/np.sum(ag.belief)
        b=ag.belief.copy()
        b=ag.perseus_belief(b)
        error_exceeded=True
        if min_error is not None:
            error_exceeded=False
            error=vf.bellmanErrorNew(b)
            if error>min_error:
                error_exceeded=True
            if error<0:
                print("negative error ",error)

        if b.tobytes() not in new_belief_dict and error_exceeded:
            new_belief_dict[b.tobytes()]=1
            new_beliefs.append(b.copy())
            if ordered:
                dist=abs(x0-xstart)+abs(y0-ystart)
                distances.append(dist)
            n_b+=1
        num_same=0
        for i in range(tmax):
            b=ag.stepInTime()
            env.stepInTime()


            if ag.true_pos[0]==env.x0 and ag.true_pos[1]==env.y0:
                print('found source at t=',i)
                break
            error_exceeded=True
            if min_error is not None:
                error_exceeded=False
                error=vf.bellmanErrorNew(b)
                if error>min_error:
                    error_exceeded=True
                if error<0:
                    print("negative error ",error)
            if b.tobytes() not in belief_dict and error_exceeded:
                new_belief_dict[b.tobytes()]=1
                new_beliefs.append(b.copy())
                if ordered:
                    dist=abs(x0-ag.true_pos[0])+abs(y0-ag.true_pos[1])
                    distances.append(dist)
                n_b+=1
            # make sure the agent doesn't get stuck
            if np.array_equal(ag.last_action,np.array([0,0])):
                num_same+=1
            else:
                num_same=0
            if num_same==5:
                break
            #terminate when enough beliefs have been collected
            if n_b>=n_collect:
                break
    if ordered:
        return new_beliefs,new_belief_dict,distances
    return new_beliefs,new_belief_dict


def in_cone(env,r_ag,r0,tol=1e-3):
    return env.likelihood[r_ag[0]-r0[0],r_ag[1]-r0[1]]>tol


start=timer()
gamma=float(os.environ.get('GAMMA'))
ic=int(os.environ.get('IC'))
rate=float(os.environ.get('RATE'))
shaping_factor=float(os.environ.get('SHAPING_FACTOR'))
shaping_power=float(os.environ.get('SHAPING_POWER'))
save_pol=False

if ic==1:
    hit_at_first=True
    wait=None

elif ic==2:
    hit_at_first=False
    wait=1000
elif ic==3:
    hit_at_first=False
    wait=0
else:
    print('bad initial condition choice')
    sys.exit()

print("using IC ",ic)
print("gamma=",gamma)
print("rate=",rate)
print("shaping factor=",shaping_factor)
print("shaping_power=",shaping_power)

random.seed(datetime.now())
penalty=0
shaping=shaping_factor
update_shaping_every=np.inf

x0=10
y0=20
boxscale=float(os.environ.get("BOXSCALE"))

Lx=160*boxscale
Ly=80*boxscale

gridscale=float(os.environ.get("GRIDSCALE"))
dims=(int(80*gridscale+1),int(40*gridscale+1))

x0=int(x0*gridscale)
y0=int(y0*gridscale)
dt=1/gridscale

env_params={
  "Lx": Lx,
  "Ly": Ly,
  "dims": dims,
  "x0": x0,
  "y0": y0,
  "D": 1,
  "V": 1,
  "agent_size": 1,
  "tau": 150,
  "R": rate,
  "dt":dt,
  "gamma":gamma,
  "exit_penalty": penalty,
  "time_reward":False,
  "easy_likelihood":False,
  "Uoverv":10,
  "shaping_factor":shaping_factor,
  "shaping_power":shaping_power,
  "entropy_factor":0

}

min_improve=0
del_beliefs=False
prune=False
threshold=2
ordered=False
random_walk=0
#wait=1000
prioritized=True
no_nn=False
goodtime=250

#create random agent
env=environment.SimpleEnv2D(env_params)
ag=agent.MosquitoAgent(env,np.array([-1,0]))
pol=policy.RandomPolicy2D()

#v0=-np.inf
#a0=None
#i=0
#for reward in env.rewards:
#    tmp=np.amin(reward)
#    if tmp>v0:
#        v0=tmp
#        a0=env.actions[i]
#    i+=1
vf=perseus_redux.ValueFunction(env,beliefs=None,v0=0,a0=np.array([-1,0]),ordered=ordered)

tmax=2500
NUM_BELIEFS=int(os.environ.get('NUM_BELIEFS'))
nb=NUM_BELIEFS
n_starts=100
n_trials=200
parallel=True

beliefs=[]
belief_dict={}
rewards=[]
distances=[]

test_min=0.0015*rate*dt/0.5
test_max=0.005*rate*dt/0.5
#test_min=0.05
#test_max=0.2


# ag.set_policy(pol)
# print("generating beliefs using random policy")
# beliefs,belief_dict,distances=generate_beliefs(ag,env,tmax,nb,x0=x0,y0=y0,beliefs=beliefs,belief_dict=belief_dict,hit_at_first=True,distances=distances,ordered=True,random_walk=random_walk,no_nearest_neighbors=no_nn,wait=wait)
# nb=5000
#
# beliefs,belief_dict,distances=generate_beliefs(ag,env,tmax,nb,x0=x0,y0=y0,beliefs=beliefs,belief_dict=belief_dict,close=True,distances=distances,ordered=True,random_walk=random_walk,no_nearest_neighbors=no_nn,wait=wait)

#pol=policy.InfotacticPolicy(ag,poisson=True)
pol=policy.SpaceAwareInfotaxis(ag)
#pol=policy.QMDPPolicy(ag)
#pol=policy.ThompsonSampling(ag,persistence_time=10)
ag.set_policy(pol)
thompson=False

random.seed(420)
starts=[]
deez=[]
for i in range(n_starts):

    xstart=random.randrange(env.dims[0])
    ystart=random.randrange(env.dims[1])
    while not in_cone(env,[xstart,ystart],[x0,y0],tol=test_min) or in_cone(env,[xstart,ystart],[x0,y0],tol=test_max): #or [xstart,ystart] in starts:
        xstart=random.randrange(env.dims[0])
        ystart=random.randrange(env.dims[1])
    starts.append([xstart,ystart])
    deez.append(np.abs(xstart-x0)+np.abs(ystart-y0))

av_opt_dist=np.mean(deez)
std_opt_dist=np.std(deez)

starts_2=[[35,16],[45,16],[55,16],[65,16]]
#starts_2=[[15,4],[25,4],[25,2]]

conditioned_trials=True

info_rewards=None
info_times=None
infotaxis_av_time=None
infotaxis_av_reward=None
infotaxis_std_time=None
infotaxis_std_reward=None

test_infotaxis=True
if test_infotaxis:
    print('testing space-aware infotaxis')
    info_rewards_1,info_times_1,infotaxis_av_time_1,infotaxis_av_reward_1,infotaxis_std_time_1,infotaxis_std_reward_1,info_norm_av_1,info_norm_std_1,info_failure_1,errors_1,foo,bar,baz=policy_trials(ag,env,starts,10,tmax=1000,wait=wait,no_nn=no_nn,thompson=thompson)
    if conditioned_trials:
        info_rewards,info_times,infotaxis_av_time,infotaxis_av_reward,infotaxis_std_time,infotaxis_std_reward,info_norm_av,info_norm_std,info_failure,errors,foo,bar,baz=policy_trials(ag,env,starts_2,1000,tmax=1000,wait=wait,no_nn=no_nn,thompson=thompson)
else:
    infotaxis_av_reward=None
    infotaxis_av_time=None
    infotaxis_std_reward=None
    infotaxis_std_time=None

random.seed(6969)

pol=policy.InfotacticPolicy(ag,poisson=True)
#pol.eps=0.1
print('generating beliefs through infotaxis')
beliefs,belief_dict,distances=generate_beliefs(ag,env,tmax,nb,x0=x0,y0=y0,beliefs=beliefs,belief_dict=belief_dict,hit_at_first=hit_at_first,distances=distances,ordered=True,random_walk=random_walk,no_nearest_neighbors=no_nn,wait=wait,test_min=test_min,test_max=test_max)
# vf_cs=perseus_redux.ValueFunction(env,beliefs=None,v0=0,a0=np.array([1,0]),ordered=ordered)
# vf_cs.load('infotaxis_ordered_16k_10_its')
#
# ag.set_policy(policy.OptimalPolicy(vf_cs,ag,parallel=parallel,epsilon=0))
# print('generating beliefs using cast-and-search policy (10% random)')
# beliefs,belief_dict,distances=generate_beliefs(ag,env,tmax,nb,x0=x0,y0=y0,beliefs=beliefs,belief_dict=belief_dict,hit_at_first=hit_at_first,distances=distances,ordered=True,random_walk=random_walk,no_nearest_neighbors=no_nn,wait=wait,test_min=test_min,test_max=test_max)

n_its_initial=100
n_alphas=[]
av_rewards=[]
average_times=[]
std_rewards=[]
std_times=[]
max_bellman_errors=[]
mean_bellman_errors=[]
failures=[]
norm_avs=[]
norm_stds=[]
all_times=[]
all_times_2=[]
all_rewards_2=[]
all_errors=[]
rms_bellman_errors=[]


tmax=1000
vf.add_beliefs(beliefs,distances)
del(beliefs)
del(distances)
#del(belief_dict)
best_av=np.inf
no_improve=0
patience=100

#best_time=infotaxis_av_time
best_time=0


tmax=1000
nb=10000
max_beliefs=-1

epsilon=float(os.environ.get('EPSILON'))
max_it=np.inf
from matplotlib import pyplot as plt
#vf.initiate_clusters()
max_frac_tol=np.inf

end=timer()
init_time=end-start
print('initialization done after:',init_time)
print('starting main loop')

it_times=[]
trial_times=[]
action_times=[]
perseus_times=[]
error_times=[]
backup_times=[]
bprime_times=[]
array_times=[]
v_update_times=[]
save_times=[]

for it in range(n_its_initial):
    start=timer()
    if it==max_it:
        max_frac_tol=0.1
    perseus_start=timer()
    vf.compute_taus_probs()
    backedup,max_bellman,mean_bellman,rms_bellman,error_time,backup_time,bprime_time,array_time,v_update_time=vf.computeOptimal(maxit=1,minit=1,verbose=True,max_frac_tol=max_frac_tol,parallel=parallel,v_tol=1e-12,prioritized=prioritized,min_improve=min_improve,boltzmann=False)
    perseus_end=timer()
    perseus_time=perseus_end-perseus_start
    print("perseus iteration done after",perseus_time)
    n_alphas.append(len(vf.alphas))
    print(len(vf.alphas),' alphas')
    max_bellman_errors.append(max_bellman)
    mean_bellman_errors.append(mean_bellman)
    rms_bellman_errors.append(rms_bellman)
    ag.set_policy(policy.OptimalPolicy(vf,ag,parallel=parallel,epsilon=epsilon))
    # for b in backedup:
    #     plt.imshow(b.data)
    #     plt.title('dist='+str(b.dist))
    #     plt.colorbar()
    #     plt.show()
    if len(vf.alphas)>=4:
        if prune:
            ag.policy.set_used=True
        rewards,times,av_time,av_reward,std_time,std_reward,norm_av,norm_std,failure,errors,trial_time_1,action_time_1,error_time_1=policy_trials(ag,env,starts,10,tmax=tmax,errors=True,vf=vf,wait=wait,no_nn=no_nn)
        if conditioned_trials:
            rewards_2,times_2,av_time_2,av_reward_2,std_time_2,std_reward_2,norm_av_2,norm_std_2,failure_2,poop,trial_time_2,action_time_2,error_time_2=policy_trials(ag,env,starts_2,200,tmax=tmax,wait=wait,no_nn=no_nn)
        #trial_time=trial_time_1+trial_time_2
        #action_time=action_time_1+action_time_2
        #print("trials done after",trial_time,"of which",action_time,"was agent timestepping and",error_time_1+error_time_2,"was Bellman error computation")
        if prune and len(vf.alphas)>=4:
            vf.prune_unused_alphas(fraction=0,threshold=threshold)
            ag.policy.set_used=False
    else:
        av_time=tmax
        av_reward=0
        std_time=0
        std_reward=0
        failure=1
        norm_av=None
        norm_std=None
        times=None
        rewards_2=None
        times_2=None
        errors=None
        trial_time=0
        action_time=0
    if times is not None:
        if conditioned_trials:
            for j in range(len(starts_2)):
                print('average arrival time from',starts_2[j],'is',np.mean(times_2[j,:]),'+/-',np.std(times_2[j,:])/np.sqrt(n_trials))
                print('(compare to',np.mean(info_times[j,:]),'+/-',np.std(info_times[j,:])/np.sqrt(n_trials),' for infotaxis)')
                print('(compare to',np.abs(starts_2[j][0]-x0)+np.abs(starts_2[j][1]-y0),'for MDP optimum)')
    #print('failure rate is ',failure)
    #print('normalized average time is',norm_av,'+/-',norm_std)
    #if test_infotaxis:
    #    print('(compare to ',infotaxis_norm_av,'+/-',infotaxis_norm_std,' for infotaxis)')
        print('average time is ',av_time,'+/-',std_time)
    #if test_infotaxis:
    #    print('(compare to ',infotaxis_av_time,'+/-',infotaxis_std_time,' for infotaxis)')
        print('average reward is ',av_reward,'+/-',std_reward)
        print('rms bellman error during trials was',np.sqrt(np.mean(np.array(errors)**2)))
    #if test_infotaxis:
    #    print('(compare to ',infotaxis_av_reward,'+/-',infotaxis_std_reward,' for infotaxis)')
    #print('optimum time is ',av_opt_dist,'+/-',std_opt_dist)
    #print('optimum reward is ',np.mean(gamma**(np.array(deez)-1)),'+/-',np.std(gamma**(np.array(deez)-1)))
    av_rewards.append(av_reward)
    average_times.append(av_time)
    std_rewards.append(std_reward)
    std_times.append(std_time)
    failures.append(failure)
    norm_avs.append(norm_av)
    norm_stds.append(norm_std)
    all_times.append(times)
    if conditioned_trials:
        all_times_2.append(times_2)
        all_rewards_2.append(rewards_2)
    all_errors.append(errors)
    dmax=8
    for d in range(1,dmax):
        v_ave=0
        n=0
        for i in range(0,d+1):
            for j in range(0,d+1):
                if np.abs(i)+np.abs(j)==d:
                    b=np.zeros_like(env.likelihood)
                    b[i,j]=1
                    v,poop=vf.value(b,parallel=True)
                    v_ave+=v
                    n+=1
        v_ave=v_ave/n
        print("Average estimated value of belief sharply peaked at distance ",d,"from source is ",v_ave)
        if shaping_power==0:
            print("(should be ",gamma**(d-1)+shaping*np.log(d)+1,")")
        else:
            print("(should be ",gamma**(d-1)+shaping*d**shaping_power,")")

    data={
    "average_times": average_times.copy(),
    "average_rewards": av_rewards.copy(),
    "std_times": std_times,
    "std_rewards": std_rewards,
    "changed_beliefs": vf.changed_beliefs.copy(),
    "maxvalues": vf.maxvalue_history.copy(),
    "num_alpha":n_alphas.copy(),
    "infotaxis_av_reward":infotaxis_av_reward,
    "infotaxis_av_time":infotaxis_av_time,
    "infotaxis_std_reward": infotaxis_std_reward,
    "infotaxis_std_time":infotaxis_std_time,
    "optimum_times":deez,
    "max_bellman":max_bellman_errors,
    "mean_bellman":mean_bellman_errors,
    "failure":failures,
    "times":all_times,
    "norm_avs":norm_avs,
    "norm_stds":norm_stds,
    "starts":starts,
    "infotaxis_times":info_times,
    "infotaxis_times_1":info_times_1,
    "infotaxis_rewards_1":info_rewards_1,
    "rewards_2":all_rewards_2,
    "times_2":all_times_2,
    "errors":all_errors,
    "rms_bellman":rms_bellman_errors
    }

    with open('convergence_rate_'+str(rate)+'_gamma_'+str(gamma)+'_ic'+str(ic)+'_shaping_factor_'+str(shaping_factor)+'_shaping_power_'+str(shaping_power)+'_nb_'+str(NUM_BELIEFS)+'_epsilon_'+str(epsilon)+'_gridscale_'+str(gridscale)+'_boxscale_'+str(boxscale)+'.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
    save_start=timer()
    if save_pol:
        vf.save('vf_rate_'+str(rate)+'_gamma_'+str(gamma)+'_ic'+str(ic)+'_it_'+str(it)+'_shaping_factor_'+str(shaping_factor)+'_shaping_power_'+str(shaping_power)+'_nb_'+str(NUM_BELIEFS)+'_epsilon_'+str(epsilon)+'_gridscale_'+str(gridscale)+'_boxscale_'+str(boxscale),no_beliefs=True)
    #vf.save('vf_',no_beliefs=True)
    save_end=timer()
    save_time=save_end-save_start
    if av_time<best_av:
        best_av=av_time
        no_improve=0
    elif len(vf.alphas)>50:
        no_improve+=1
    if no_improve==patience:
        break
    if av_time<best_time and len(vf.beliefs)<max_beliefs:
        best_time=av_time
        print("augmenting beliefs...")
        if del_beliefs:
            vf.beliefs=[]
        #ag.set_policy(policy.OptimalPolicy(vf,ag,parallel=parallel))
        beliefs=[]
        distances=[]

        beliefs,belief_dict,distances=generate_beliefs(ag,env,2500,1000,x0=x0,y0=y0,beliefs=beliefs,belief_dict=belief_dict,hit_at_first=hit_at_first,distances=distances,ordered=True,random_walk=random_walk,no_nearest_neighbors=no_nn,wait=wait,test_min=test_min,test_max=test_max,min_error=np.mean(np.abs(errors)),vf=vf)
        vf.add_beliefs(beliefs,distances)
        # with open('beliefs_'+str(i+1)+'_its.pkl', 'wb') as f:
        #     pickle.dump(beliefs, f, pickle.HIGHEST_PROTOCOL)
        # with open('distances_'+str(i+1)+'_its.pkl', 'wb') as f:
        #     pickle.dump(distances, f, pickle.HIGHEST_PROTOCOL)
        del(beliefs)
        del(distances)
    if (it+1)%update_shaping_every==0 and shaping>0:
        shaping=shaping*0.5
        if shaping<1e-5:
           shaping=0
        env.change_shaping_factor(shaping,vf)
    if it>=max_it:
        max_frac_tol=max_frac_tol*0.5
    
    end=timer()
    it_time=end-start
    print('iteration done after',it_time)
    #it_times.append(it_time)
    #trial_times.append(trial_time)
    #action_times.append(action_time)
    #perseus_times.append(perseus_time)
    #error_times.append(error_time)
    #backup_times.append(backup_time)
    #bprime_times.append(bprime_time)
    #array_times.append(array_time)
    #v_update_times.append(v_update_time)
    #save_times.append(save_time)
    #benchmarks={
    #"init_time":init_time,
    #"it_times":it_times,
    #"trial_times":trial_times,
    #"action_times":action_times,
    #"perseus_times":perseus_times,
    #"error_times":error_times,
    #"backup_times":backup_times,
    #"bprime_times":bprime_times,
    #"array_times":array_times,
    #"v_update_times":v_update_times,
    #"save_times":save_times
    #}
    #with open('benchmarks_rate_'+str(rate)+'_gamma_'+str(gamma)+'_ic'+str(ic)+'_shaping_factor_'+str(shaping_factor)+'_shaping_power_'+str(shaping_power)+'_nb_'+str(NUM_BELIEFS)+'_epsilon_'+str(epsilon)+'_v2.pkl', 'wb') as f:
     #   pickle.dump(benchmarks, f, pickle.HIGHEST_PROTOCOL)
