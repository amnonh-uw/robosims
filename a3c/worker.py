import numpy as np
import tensorflow as tf
import scipy.signal
import a3c.helper
import random
import math

# ### Worker Agent

class Worker():
    def __init__(self,name, game, args, ac_network, trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = ac_network(args,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)
        self.update_feedforward_local_ops = update_target_graph('global', self.name, tf.GraphKeys.MODEL_VARIABLES)

        self.env = game(args)
        
    def train(self,rollout,sess,gamma,bootstrap_value):
        print("Worker training...")
        rollout = np.array(rollout)
        t_inputs = rollout[:,0]
        s_inputs = rollout[:, 1]
        sensor_inputs = rollout[:, 2]
        actions = rollout[:,3]
        rewards = rollout[:,4]
        values = rollout[:,5]
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)



        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.t_input:np.stack(t_inputs, axis=0),
            self.local_AC.s_input:np.stack(s_inputs, axis=0),
            self.local_AC.sensor_input:np.stack(sensor_inputs, axis=0),
            self.local_AC.actions:np.stack(actions, axis=0),
            self.local_AC.advantages:advantages}


        r_o, v_l,p_l,e_l,g_n,v_n,_ = sess.run([
            self.local_AC.responsible_outputs,
            self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)


        for i in range(s_inputs.shape[0]):
            print("actions[{}]={}".format(i, actions[i]))
            print("advantages[{}]={}".format(i, advantages[i]))
            print("responsible outputs[{}]={}".format(i, r_o[i]))
            print("discounted_rewards[{}]={}".format(i, discounted_rewards[i]))
            print("value_plus[{}]={}".format(i, self.value_plus[i]))

        print("losses: policy {} value {} entropy {}".format(p_l, v_l, e_l))
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
        
    def work(self,args,sess,coord,saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))

        with sess.as_default(), sess.graph.as_default():
            # copy non trainable variables from global network
            sess.run(self.update_feedforward_local_ops)
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_source_frames = []
                episode_sensor_inputs = []
                episode_reward = 0
                episode_step_count = 0
                
                self.env.new_episode()
                t = self.env.get_state().target_buffer()
                s = self.env.get_state().source_buffer()
                sensor_input = self.env.get_state().sensor_input()
                episode_source_frames.append(s)
                episode_target_frame = t
                episode_sensor_inputs.append(sensor_input)

                t_input = process_frame(t)
                s_input = process_frame(s)
                
                while self.env.is_episode_finished() == False:
                    if args.discrete_actions:
                        #Take an action using probabilities from policy network output.
                        a_dist,v = sess.run([self.local_AC.policy,self.local_AC.value],
                            feed_dict={self.local_AC.s_input:[s_input],
                                        self.local_AC.t_input:[t_input],
                                        self.local_AC.sensor_input:[sensor_input]})

                        a = np.random.choice(a_dist[0],p=a_dist[0])
                        a = np.argmax(a_dist == a)

                        r = self.env.take_discrete_action(a) / 100.0
                    else:
                        #Take an action using random gaussian from the policy network output
                        means, variances, v = sess.run([self.local_AC.policy_means,self.local_AC.policy_variances, self.local_AC.value],
                            feed_dict={self.local_AC.s_input:[s_input],
                                        self.local_AC.t_input:[t_input],
                                        self.local_AC.sensor_input:[sensor_input]})

                        print("means: {}".format( means[0, :]))
                        print("variances: {}".format(variances[0, :]))
                        x = random.normalvariate(means[0, 0], math.sqrt(variances[0, 0]))
                        y = random.normalvariate(means[0, 1], math.sqrt(variances[0, 1]))
                        z = random.normalvariate(means[0, 2], math.sqrt(variances[0, 2]))
                        ry = random.normalvariate(means[0, 3], math.sqrt(variances[0, 3]))
                        a = [x, y, z, ry]

                        r = self.env.take_continous_action(x, y, z, ry) / 100.0

                    d = self.env.is_episode_finished()
                    if d == False:
                        new_s = self.env.get_state().source_buffer()
                        new_sensor_input = self.env.get_state().sensor_input()

                        episode_source_frames.append(new_s)
                        episode_sensor_inputs.append(new_sensor_input)

                        new_s_input = process_frame(new_s)
                    else:
                        new_s_input = s_input
                        new_sensor_input = sensor_input

                    episode_buffer.append([t_input, s_input, sensor_input, a, r, v[0,0]])
                    episode_values.append(v[0,0])
                    episode_sensor_inputs.append(sensor_input)

                    episode_reward += r

                    s_input = new_s_input
                    sensor_input = new_sensor_input

                    total_steps += 1
                    episode_step_count += 1
                    
                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == args.episode_buffer_size and d != True and episode_step_count != args.max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.s_input:[s_input],
                                   self.local_AC.t_input:[t_input],
                                   self.local_AC.sensor_input:[sensor_input]})[0,0]
                        v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,args.gamma,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break
                                            
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,args.gamma,0.0)
                                
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_source_frames)
                        a3c.helper.make_gif(images,'./frames/image'+str(episode_count)+'.gif',
                            duration=len(images)*time_per_step)
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope, graphkeys = tf.GraphKeys.TRAINABLE_VARIABLES):
    from_vars = tf.get_collection(graphkeys, from_scope)
    to_vars = tf.get_collection(graphkeys, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def process_frame(frame):
    return frame.astype(float)/ 255.0

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

