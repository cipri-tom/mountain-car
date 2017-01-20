import sys

import pylab as plb
import numpy as np
import mountaincar

def softmax(x, tau):
    """ Returns softmax probabilities with temperature tau"""
    # TODO: if overflows or unstable, use normalisation trick
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()

class Agent():
    """A Sarsa(lambda) agent which learns its way out """

    def __init__(self, mountain_car = None, tau=1, weights = None,
                 side_size = 20, x_range = (-150, 30), v_range = (-15, 15)):
        """ Makes a new agent with given parameters:
            mountain_car : Instance of MountainCar
            tau          : exploration temperature -- scalar
            side_size    : input neurons are arranged in a grid of this size -- scalar
            x_range      : range of positions to with input neurons -- 2-tuple
            v_range      : range of velocities to cover with input neurons -- 2-tuple
            weights      : from input neurons to output neurons -- array(3 x side_size x side_size)
        """

        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car


        self.eligibility_trace = np.zeros((3,input_side_size*input_side_size))

        if weights is None:
            self.weights = np.ones((3,side_size,side_size))
            #self.weights = np.random.rand(3,side_size*side_size)
        else:
            self.weights = weights

        self.tau = tau

        # neuron preference centres, widths, activations:
        self.centres_x, self.sigma_x = np.linspace(x_range[0], x_range[1], side_size-1, endpoint=True, retstep=True)
        self.centres_v, self.sigma_v = np.linspace(v_range[0], v_range[1], side_size-1, endpoint=True, retstep=True)
        self.states = np.zeros((side_size, side_size))

        self.side_size  = side_size

    def visualize_trial(self, n_steps = 2000):
        """ Do an episode of maximum `n_steps` """

        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()

        # make sure the mountain-car is reset
        #self.mountain_car.reset()

        Qs, Qss = 0,0
        action, input_activities = 0,[]

        self.eligibility_trace = np.zeros((3,self.input_side_size*self.input_side_size))

        for n in range(n_steps):
            #print('\rt =', self.mountain_car.t)
            sys.stdout.flush()

            # choose action
            x = self.mountain_car.x
            v = self.mountain_car.x_d

            if n==0:
                action, Qs, input_activities = self.choose_action(x, v)

            #car update
            self.mountain_car.apply_force(action)
            self.mountain_car.simulate_timesteps(100, 0.01)


            #s' state
            x2 = self.mountain_car.x
            v2 = self.mountain_car.x_d

            action2,Qs2,input_activities2 = self.choose_action(x2, xdot2,tau)

            #learn
            self.learn(action, Qs, Qs2, input_activities)

            #next state
            Qs = Qs2
            action = action2
            input_activities = input_activities2


            # update the visualization
            mv.update_figure()
            plb.draw()


            # check for rewards
            if self.mountain_car.R > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break

    def visualize_field(self):
        actions = np.zeros((self.input_side_size,self.input_side_size))

        sigma = (self.x_range[1]-self.x_range[0])/(self.input_side_size-1)
        sigmadot = (self.xdot_range[1]-self.xdot_range[0])/(self.input_side_size-1)

        for i in range(self.input_side_size):
            for j in range(self.input_side_size):

                x = self.x_range[0]+j*sigma
                xdot = self.xdot_range[0]+i*sigmadot

                input_activities = np.zeros(self.input_side_size*self.input_side_size)

                for i0 in range(self.input_side_size):
                    for j0 in range(self.input_side_size):

                        x0 = self.x_range[0]+j0*sigma
                        xdot0 = self.xdot_range[0]+i0*sigmadot

                        input_activities[i*self.input_side_size+j] = np.exp(-(x-x0)**2/(sigma)**2-(xdot-xdot0)**2/(sigmadot)**2)

                output_activities = np.zeros(3)
                for k in range(3):
                    output_activities[k] = self.outputs_weights[k].dot(input_activities)

                actions[i,j] = np.argmax(output_activities)

        print(actions)

   def activation(self, i, j, x, v):
        """ The activation of the neuron at position (i,j) given (x, v) """
        xc, vc = self.centres_x[i], self.centres_v[j]

        return np.exp(-(x-xc)**2 / (self.sigma_x)**2
                      -(v-vc)**2 / (self.sigma_v)**2)


    def choose_action(self, x, v):
        """ Computes the action from state (x,v) following an ε-greedy strategy.
            It uses a linear approximation of the value function and softmax
            probability distribution.

            Return: (action_prime, Q[(x,v),action_prime], s_prime)
        """

        # 1) find Q of each possible action from state s=(x,v)
        # This is approximated linearly from `s` and each action's 'feature vector'

        # We consider that each action is described by a set of features, wohse weights are learnt
        # We use a tiling of neurons over the whole state space to find a linear approximation of Q at s=(x,v)
        # given the weights `w` of some action's features

        # 1.1) Map s=(x,v) to neurons based on the activation function
        # TODO: vectorize this (using meshgrid)
        s_prime = np.zeros_like(self.states) # in_activations
        for i in range(self.side_size):
            for j in range(self.side_size):
                s_prime[i,j] = self.activation(i,j, x,v)

        # 1.2) For each action `a_i`, compute Q(s, a_i) -- Vectorised
        # weights(num_actions x num_neurons_pos x num_neurons_v) . states(neuron_activ_pos x neuron_activ_v) = (num_Q_actions)
        # i.e. (3x20x20) . (20x20) -> (3,)
        # out_activations = np.tensordot(self.weights, s_prime, 2)
        Q_sa = np.tensordot(self.weights, s_prime, 2)

        # ε-greedy action choice; Generate action probability with `tau` exploration temperature
        action_probabilities = softmax(Q_sa, tau)
        action_prime = np.random.choice([-1,0,1], p=action_probabilities)

        return action_prime, Q_sa[action_prime+1], s_prime

    def learn(self, action, Qs, Qss, input_activities, eta = .01, gamma = .99, lambdaa = 0.90):
        TD = self.mountain_car.R - (Qs - gamma*Qss)
        #print(TD)
        for i in range(self.input_side_size):
            for j in range(self.input_side_size):
                for k in [0,1,2]:
                    self.eligibility_trace[k,i*self.input_side_size+j] *= gamma * lambdaa

                self.eligibility_trace[action+1,i*self.input_side_size+j] += input_activities[i*self.input_side_size+j] #action belongs to {-1,0,1}
                for k in [0,1,2]:
                    self.outputs_weights[k,i*self.input_side_size+j] += eta * TD * self.eligibility_trace[k,i*self.input_side_size+j]

