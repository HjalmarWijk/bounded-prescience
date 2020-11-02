import gym
import numpy as np
import time

import lucid
from lucid.modelzoo.vision_base import Model
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.transform as transform
import lucid.optvis.render as render
import tensorflow as tf

from atari_zoo import MakeAtariModel
from lucid.optvis.render import import_model

from atver.labelling import get_property

from atver import get_wrapped

def check_noops(env_str,method_str,prop_str,max_noops=500,render = False,max_frames = 100000,verbose = False,path='../models',runid = 0,min_noops=0):
    labeller = get_property(env,prop_str)
    model_name_map = {
            'A2C': 'a2c',
            'DQN-D': 'dqn',
            'Rainbow-D': 'rainbow',
            'IMPALA-U':'impala',
            'APEX': 'apex'
            }
    m = MakeAtariModel(model_name_map[method_str],env_str,runid,tag='final')()
    m.load_graphdef()
    config = tf.ConfigProto(
            device_count = {'GPU': 1}
        )
    config.gpu_options.allow_growth=True
    violation_list = []
    noop_violation_list = []
    reward_list = []
    with tf.Graph().as_default() as graph, tf.Session(config=config) as sess:
        env = get_wrapped(env_str,method_str,max_frames)

        nA = env.action_space.n
        X_t = tf.placeholder(tf.float32, [None] + list(env.observation_space.shape))

        T = import_model(m,X_t,X_t)
        policy = T(m.layers[-1]['name'])
        order = tf.argsort(-:qpolicy,axis=1)
        
        

        if verbose:
            print('Checking if %s ever violates %s in %s' % (method_str, prop_str, env_str))
        for i in range(min_noops,max_noops):
            tot_reward = 0
            violation_steps = 0
            violation = False
            obs = env.reset()[0]
            trace = []
            done = False
            if verbose:
                print('Running with %d noops' % i)
            for j in range(i):
                trace.append(0)
                mod,orig = env.step(0)
                obs_orig, reward, done, info = orig
                obs = mod[0]
                tot_reward += reward
                if labeller.label(obs_orig,reward,done,info):
                    if not violation:
                        if verbose:
                            print('Property violated during noops at step %d' % violation_steps)
                        violation_list.append(violation_steps)
                        violation = True
                    elif render:
                        print('Property violated during noops at step %d' % violation_steps)
                if done:
                    break
                violation_steps += 1
            if not violation:
                noop_violation_list.append("None")
            violation = False
            violation_steps = 0
            if not done:
                for j in range(100000):
                    if render:
                        env.render()
                    train_dict = {X_t:obs[None]}

                    results = sess.run([action_sample], feed_dict=train_dict)
                    #grab action
                    action = np.squeeze(results[0])
                    trace.append(action)
                    mod,orig = env.step(action)
                    obs_orig, reward, done, info = orig
                    obs = mod[0]
                    tot_reward += reward
                    if labeller.label(obs_orig,reward,done,info):
                        if not violation:
                            if verbose:
                                print('Property violated at step %d' % violation_steps)
                            violation_list.append(violation_steps)
                            violation = True
                    if done:
                        break
                    violation_steps += 1
            if not violation:
                violation_list.append("None")
            reward_list.append(tot_reward)
    return violation_list, noop_violation_list, reward_list
