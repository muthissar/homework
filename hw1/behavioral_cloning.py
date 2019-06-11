import tensorflow as tf
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.python.data.ops.iterator_ops import Iterator
from itertools import count
from run_expert import run_expert
from gym import wrappers
from mujoco_py.generated import const
import glfw
def get_dataset(expert_data_file=None,envname=None,render=False,expert_policy_file=None,max_timesteps=None,num_rollouts=None,store=False,batch_size=50):
    returns = None
    if expert_data_file:
        with open(expert_data_file, 'rb') as f:
            trace = pickle.load(f)
            returns = trace['returns']
    else:
        returns, trace = run_expert(envname,render,expert_policy_file,max_timesteps,num_rollouts,store=store)
    dataset = tf.data.Dataset.from_tensor_slices((trace['observations'], trace['actions']))
    dataset = dataset.shuffle(buffer_size=10000)
    return dataset, returns

class BehavioralCloningNet():
    def __init__(self, obs_dim, action_dim, batch_size, learning_rate=0.001, scope="behavioral_cloning"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(dtype=tf.float32, shape=[None] + obs_dim, name='state')
            self.target = tf.placeholder(dtype=tf.float32, shape=[None] + action_dim, name="target")
            n_hidden = 50
            self.dropout = tf.keras.layers.Dropout(rate=0.5)
            self.hidden_layer_1 = tf.contrib.layers.fully_connected(
                inputs = self.dropout(self.state),
                num_outputs=n_hidden,
            ) 
            #self.hidden_layer_2 = tf.contrib.layers.fully_connected(
            #    inputs = self.dropout(self.hidden_layer_1),
            #    num_outputs=n_hidden,
            #    weights_initializer=tf.contrib.layers.xavier_initializer()
            #) 
            
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs = self.dropout(self.hidden_layer_1),
                num_outputs=action_dim[0], #predict all actions
                activation_fn=None,
            )

            
            self.loss = tf.losses.mean_squared_error(predictions=self.output_layer,labels=self.target)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())
            tf.get_default_session().run(tf.global_variables_initializer())
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        sess.run(self.output_layer, { self.state: state })
        return sess.run(self.output_layer, { self.state: state })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        #TODO: remove numpy stuff
        #state = tf.ensure_shape(state,self.state.shape.as_list()).eval()
        feed_dict = { self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def train(self, next_element: Iterator):
        sess = tf.get_default_session()
        losses = []
        for i in count():
            try:
                state_val, target_val = sess.run(next_element)
                loss_val = self.update(state_val,np.squeeze(target_val))
                losses.append(loss_val)
            except tf.errors.OutOfRangeError:
                break
        return losses
    

class Evaluate():
    #TODO: deprecate render only use in evaluate
    def __init__(self,env,max_steps = None,render=False,video_path=None):
        self.env = env
        self.render = render
        import gym
        self.env = gym.make(env)
        if max_steps is None:
            self.max_steps = self.env.spec.timestep_limit
        #if video_path:
        #    self.env = wrappers.Monitor(self.env, video_path, force=True) # record the game as as an mp4 file
    def step(self,obs) -> list:
        raise 'Not Implimented'

    def evaluate(self,rollouts=1,render=False, video_path=None):
        returns = []
        observations = []
        actions = []
        
        for i in range(rollouts):
            # Animate first rollout if asked to do so.
            old_env = self.env    
            #self.env = wrappers.Monitor(self.env, video_path, force=True) if i is 0 and\
            #    video_path  else self.env
            obs = self.env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action =  self.step(obs)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps += 1
                if render and i == 0:
                    self.env.render(mode='human')
                    viewer = self.env.env.viewer                
                    #HIDE CONTROLS
                    viewer.key_callback(None,glfw.KEY_H,None,glfw.RELEASE,None)
                    #PAUSE 
                    #viewer.key_callback(None,glfw.KEY_SPACE,None,glfw.RELEASE,None)
                    #FOLLOW AGENT
                    viewer.key_callback(None,glfw.KEY_TAB,None,glfw.RELEASE,None)
                    
                if steps >= self.max_steps:
                    self.env.close()
                    break
            returns.append(totalr)
            self.env = old_env
        return returns

class EvaluateDagger(Evaluate):
    def __init__(self,env,expert_policy,behavioral_net,max_steps = None,render=False):

        super().__init__(env,max_steps,render)
        self.expert_obs = []
        self.expert_actions = []
        self.expert_policy = expert_policy
        self.behavioral_net = behavioral_net

    def step(self,obs):
        self.expert_obs.append(obs)
        self.expert_actions.append(self.expert_policy(obs[None,:]))
        return self.behavioral_net.predict([obs])
    