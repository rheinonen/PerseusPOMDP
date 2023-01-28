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
import utils

def conditional_mean(times,tmax):
    filtered=[x for x in times if x<tmax]
    mean=np.mean(filtered)
    err=np.std(filtered)/np.sqrt(len(filtered))
    failure=1-len(filtered)/len(times)
    return mean,err,failure

def initialize_belief_and_source(ag,env,boundary_tol=1,force_obs=None,force_source=None,min_radius=None,old_init=False,test_min=0.015,test_max=0.05,t_wait=1000):
    if old_init:
        #old_pos=ag.true_pos
        x0=10
        y0=20
        env.set_pos(x0,y0)
        xstart=random.randrange(env.dims[0])
        ystart=random.randrange(env.dims[1])

        while (xstart==x0 and ystart==y0) or not utils.in_cone(env,[xstart,ystart],[x0,y0],tol=test_min) or utils.in_cone(env,[xstart,ystart],[x0,y0],tol=test_max):
            xstart=random.randrange(env.dims[0])
            ystart=random.randrange(env.dims[1])
        pol=copy(ag.policy)
        ag.set_policy(policy.TrivialPolicy())
        while True:
            ag.reset([xstart,ystart])
            for i in range(t_wait):
                ag.stepInTime()
                env.stepInTime()
                if ag.last_obs:
                    break
            if ag.last_obs:
                break
        ag.set_policy(pol)
    else:       
        if force_obs is not None:
            obs=force_obs
        else:
            obs=utils.initial_hit_aurore(env)
        ag.updateBelief(obs)
        if min_radius is not None:
            x0=ag.true_pos[0]
            y0=ag.true_pos[1]
            for i in range(-min_radius,min_radius+1):
                for j in range(-min_radius,min_radius+1):
                    if i**2+j**2<min_radius**2 and i+x0<ag.belief.shape[0] and j+y0<ag.belief.shape[1] and i+x0>=0 and j+y0>=0:
                        ag.belief[x0+i,y0+j]=0
            ag.belief/=np.sum(ag.belief)
    if force_source is not None:
        xcoord,ycoord=force_source
    else:
        index=np.random.choice(np.arange(env.dims[0]*env.dims[1],dtype='int'),p=ag.belief.flatten())
        xcoord,ycoord=np.unravel_index(index,env.dims)
        while on_boundary([xcoord,ycoord]):
            index=np.random.choice(np.arange(env.dims[0]*env.dims[1],dtype='int'),p=ag.belief.flatten())
            xcoord,ycoord=np.unravel_index(index,env.dims)
    env.set_pos(xcoord,ycoord)

def on_boundary(pos,dims=(81,41),tol=1):
    if pos[0]<=tol or pos[0]>=dims[0]-1-tol or pos[1]<=tol or pos[1]>=dims[1]-1-tol:
        return True
    return False

def policy_trials(ag,env,n_trials,thompson=False,tmax=1000,ag_start=[65,20],errors=False,vf=None,error_frac=0.3333,bad_traj=False,force_obs=None,force_source=None,min_radius=None,old_init=False,test_min=0.015,test_max=0.05,t_wait=1000):
    times=[]
    starts=[]
    hits=[]
    if bad_traj:
        bad_trajs=[]
    else:
        bad_trajs=None
    if errors:
        bellman=[]
    else:
        bellman=None
    for j in range(n_trials):
        env.reset()
        ag.reset(ag_start)
        if thompson:
            ag.policy.reset()
        initialize_belief_and_source(ag,env,force_obs=force_obs,force_source=force_source,min_radius=min_radius,old_init=old_init,test_min=test_min,test_max=test_max,t_wait=t_wait)
        starts.append([env.x0,env.y0])
        traj=[ag.true_pos]
        b0=ag.belief
        for k in range(tmax):
            if errors:
                b,v=ag.stepInTime(values=True)
            else:
                b=ag.stepInTime(make_obs=k!=0)
            if errors and random.random()<error_frac:
                bellman.append(vf.bellmanErrorNew(b,value=v))
            env.stepInTime()
            traj.append(ag.true_pos)
            if ag.true_pos[0]==env.x0 and ag.true_pos[1]==env.y0:
                hits.append(ag.nhits-1)
                break
            if ag.stuck_count>8:
                k=tmax-1
                break
        times.append(k+1)
        if k>200 and bad_traj:
            bad_trajs.append([j,traj])
    return times,starts,b0,hits,bellman,bad_trajs

def generate_beliefs(ag,env,tmax,n_collect,ag_start=[65,20],beliefs=None,belief_dict=None,force_obs=None,min_radius=None,old_init=False,test_min=0.015,test_max=0.05,t_wait=1000):
    if beliefs is None:
        new_beliefs=[]
    else:
        new_beliefs=beliefs
    if belief_dict is None:
        new_belief_dict={}
    else:
        new_belief_dict=belief_dict
    n_b=0
    runs=0
    k=1
    while n_b<n_collect:
        ag.reset(ag_start)
        env.reset()
        initialize_belief_and_source(ag,env,force_obs=force_obs,min_radius=min_radius,old_init=old_init,test_min=test_min,test_max=test_max,t_wait=t_wait)
        b=ag.belief.copy()
        b=ag.perseus_belief(b)
        if np.isnan(b).any():
            print("Warning: belief contains NaN")    
        elif b.tobytes() not in new_belief_dict:
            new_belief_dict[b.tobytes()]=1
            new_beliefs.append(b.copy())
            n_b+=1
        for i in range(tmax):
            b=ag.stepInTime(make_obs=i>0)
            env.stepInTime()
            if ag.true_pos[0]==env.x0 and ag.true_pos[1]==env.y0:
                #print('found source at t=',i)
                break
            if b.tobytes() not in new_belief_dict:
                new_belief_dict[b.tobytes()]=1
                new_beliefs.append(b.copy())
                n_b+=1
            #terminate when enough beliefs have been collected
            if n_b>=n_collect:
                break
        runs+=1
        if n_b>=1000*k:
            print(n_b,'beliefs after',runs,'runs')
            k+=1
    return new_beliefs,new_belief_dict

gamma=float(os.environ.get('GAMMA'))
rate=float(os.environ.get('RATE'))
shaping_factor=float(os.environ.get('SHAPING_FACTOR'))
shaping_power=float(os.environ.get('SHAPING_POWER'))
save_pol=True
min_radius=int(os.environ.get('MIN_RADIUS'))

print("gamma=",gamma)
print("rate=",rate)
print("shaping factor=",shaping_factor)
print("shaping_power=",shaping_power)

shaping=shaping_factor
update_shaping_every=np.inf

tmax=int(os.environ.get('TMAX'))

env_params={
  "Lx": 160,
  "Ly": 80,
  "dims":(81,41),
  "x0": 0,
  "y0": 0,
  "D": 1,
  "V": 1,
  "agent_size":1,
  "tau":150,
  "R":rate,
  "dt":1,
  "gamma":gamma,
  "exit_penalty":0,
  "time_reward":False,
  "easy_likelihood":False,
  "Uoverv":8,
  "shaping_factor":shaping,
  "shaping_power":1,
  "entropy_factor":0,
  "2d": False,
  "max_detections":1
}

prioritized=True
env=environment.SimpleEnv2D(env_params)
ag=agent.MosquitoAgent(env,np.array([0,0]))

vf=perseus_redux.ValueFunction(env,beliefs=None,v0=0,a0=np.array([-1,0]),ordered=False)

NUM_BELIEFS=int(os.environ.get('NUM_BELIEFS'))
nb=NUM_BELIEFS
n_trials=int(os.environ.get('N_TRIALS'))
parallel=True
old_init=int(os.environ.get('OLD_INIT'))
old_init=bool(old_init)

beliefs=[]
belief_dict={}
rewards=[]
distances=[]

#pol=policy.InfotacticPolicy(ag)
pol=policy.SpaceAwareInfotaxis(ag)
ag.set_policy(pol)
thompson=False


heuristic_times=None
heuristic_starts=None
hits=None

test_heuristic=bool(int(os.environ.get('TEST_SAI')))
if test_heuristic:
    print('testing heuristic')
    heuristic_times,heuristic_starts,b0,hits,tmp,bad_traj=policy_trials(ag,env,n_trials,tmax=tmax,bad_traj=False,thompson=thompson,force_obs=1,min_radius=min_radius,old_init=old_init,test_min=0.0015*rate/0.5,test_max=0.005*rate/0.5,t_wait=1000)
    mean,err,failure=conditional_mean(heuristic_times,tmax)
    print('heuristic had mean arrival time',mean,'+/-',err,'with failure rate',failure)
random.seed(6969)


data={
"heuristic_times":heuristic_times,
"heuristic_starts":heuristic_starts,
"hits":hits
}

EPSILON=float(os.environ.get('EPSILON'))

#with open('infotaxis_sanity_aurore.pkl','wb') as f:
#    pickle.dump(data,f)

with open('convergence_windy_old_init_'+str(old_init)+'_rate_'+str(rate)+'_gamma_'+str(gamma)+'_shaping_factor_'+str(shaping_factor)+'_shaping_power_'+str(shaping_power)+'_nb_'+str(NUM_BELIEFS)+'_epsilon_'+str(EPSILON)+'.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


pol=policy.InfotacticPolicy(ag,epsilon=EPSILON)
print('generating beliefs through infotaxis')
beliefs,belief_dict=generate_beliefs(ag,env,tmax,nb,beliefs=beliefs,belief_dict=belief_dict,force_obs=1,min_radius=min_radius,old_init=old_init,test_min=0.0015*rate/0.5,test_max=0.005*rate/0.5,t_wait=1000)

n_its=100
n_alphas=[]
max_bellman_errors=[]
mean_bellman_errors=[]
all_times=[]
all_errors=[]
rms_bellman_errors=[]
all_starts=[]

vf.add_beliefs(beliefs)
del(beliefs)
del(belief_dict)

max_frac_tol=np.inf

print('starting main loop')


for it in range(n_its):
    vf.compute_taus_probs()
    backedup,max_bellman,mean_bellman,rms_bellman,error_time,backup_time,bprime_time,array_time,v_update_time=vf.computeOptimal(maxit=1,minit=1,verbose=False,max_frac_tol=max_frac_tol,parallel=parallel,v_tol=1e-12,prioritized=prioritized,boltzmann=False)
    n_alphas.append(len(vf.alphas))
    print(len(vf.alphas),' alphas')
    max_bellman_errors.append(max_bellman)
    mean_bellman_errors.append(mean_bellman)
    rms_bellman_errors.append(rms_bellman)
    ag.set_policy(policy.OptimalPolicy(vf,ag,parallel=parallel))
    if len(vf.alphas)>=4:
        print("testing...")
        times,starts,b0,hits,errors,bad_trajs=policy_trials(ag,env,n_trials,tmax=tmax,errors=False,force_obs=1,min_radius=min_radius,old_init=old_init,test_min=0.0015*rate/0.5,test_max=0.005*rate/0.5,t_wait=1000)
        if test_heuristic:
            print("mean arrival time is",np.mean(times),"+/-",np.std(times)/np.sqrt(len(times)),"(compare to",np.mean(heuristic_times),"for the tested heuristic)")
        else:
            print('mean arrival time is',np.mean(times))
        print("failure rate is",np.sum(times==tmax)/n_trials)
    else:
        times=None
        errors=None
        starts=None
    all_times.append(times)
    all_starts.append(starts)
    all_errors.append(errors)
    dmax=8
    for d in range(1,dmax):
        v_ave=0
        n=0
        for i in range(0,d+1):
            for j in range(0,d+1):
                if np.abs(i)+np.abs(j)==d:
                    b=np.zeros_like(env.likelihood[0])
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
    "changed_beliefs": vf.changed_beliefs.copy(),
    "maxvalues": vf.maxvalue_history.copy(),
    "num_alpha":n_alphas.copy(),
    "max_bellman":max_bellman_errors,
    "mean_bellman":mean_bellman_errors,
    "times":all_times,
    "starts":all_starts,
    "heuristic_times":heuristic_times,
    "infotaxis_times_1":heuristic_starts,
    "errors":all_errors,
    "rms_bellman":rms_bellman_errors
    }

    with open('convergence_windy_old_init_'+str(old_init)+'_rate_'+str(rate)+'_gamma_'+str(gamma)+'_shaping_factor_'+str(shaping_factor)+'_shaping_power_'+str(shaping_power)+'_nb_'+str(NUM_BELIEFS)+'_epsilon_'+str(EPSILON)+'.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
    save_start=timer()
    if save_pol:
        vf.save('vf_windy_old_init_'+str(old_init)+'_rate_'+str(rate)+'_gamma_'+str(gamma)+'_shaping_factor_'+str(shaping_factor)+'_shaping_power_'+str(shaping_power)+'_nb_'+str(NUM_BELIEFS)+'_epsilon_'+str(EPSILON)+'_it_'+str(it),no_beliefs=True)
    #vf.save('vf_',no_beliefs=True)

    
