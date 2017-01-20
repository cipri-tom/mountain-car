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


        if weights is None:
            # TODO: is `ones` good enough ? should it be random ?
            self.weights = np.ones((3,side_size,side_size))
            #self.weights = np.random.rand((3,side_size,side_size))
        else:
            self.weights = weights

        self.eligibility_trace = np.empty_like(self.weights)

        # neuron preference centres, widths, activations:
        self.centres_x, self.sigma_x = np.linspace(x_range[0], x_range[1], side_size, endpoint=True, retstep=True)
        self.centres_v, self.sigma_v = np.linspace(v_range[0], v_range[1], side_size, endpoint=True, retstep=True)

        self.tau = tau
        self.side_size  = side_size

    def visualize_trial(self, n_steps=2000, tau=1):
        """ Do an episode of maximum `n_steps` """

        # prepare for the visualization
        # plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()

        # Initialisation:
        # ---------------
        self.mountain_car.reset()
        self.eligibility_trace = np.zeros_like(self.eligibility_trace)
        self.tau = tau

        # current state
        x = self.mountain_car.x
        v = self.mountain_car.x_d

        # representation of the current state, and current action
        action, state, Q_s_a = self.choose_action(x, v)
        Q_sp_ap = 0  # not yet known

        for n in range(n_steps):
            # Use current action to get to next state, s_prime
            self.mountain_car.apply_force(action)
            self.mountain_car.simulate_timesteps(100, 0.01)
            x_prime = self.mountain_car.x
            v_prime = self.mountain_car.x_d

            # Since this is SARSA, choose next action supposing you also use same policy in next state
            action_prime, state_prime, Q_sp_ap = self.choose_action(x_prime, v_prime)

            # update weights based on observations
            self.learn(state, action, Q_s_a, Q_sp_ap)

            # move to next state
            Q_s_a  = Q_sp_ap
            action = action_prime
            state  = state_prime

            # update the visualization
            mv.update_figure()
            # plb.draw()

            # stop when goal was reached
            if self.mountain_car.R > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break

    def visualize_field(self):
        actions = np.empty((self.side_size, self.side_size))
        state = np.empty_like(actions)

        # TODO: super vectorise this

        # for each possible state
        for i in range(self.side_size):
            # TODO: invert y axis, because higher values are lower in the matrix
            for j in range(self.side_size):

                # compute the activations of all the states, given this one
                for k in range(self.side_size):
                    for l in range(self.side_size):
                        state[k,l] = self.activation(k,l, self.centres_x[i], self.centres_v[j])

                Q_s_a = np.tensordot(self.weights, state, 2)
                actions[i,j] = np.argmax(Q_s_a) - 1  # to match true action

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

            Return: (a_prime, s_prime, , Q[s_prime,a_prime])
        """

        # 1) find Q of each possible action from state s=(x,v)
        # This is approximated linearly from `s` and each action's 'feature vector'

        # We consider that each action is described by a set of features, wohse weights are learnt
        # We use a tiling of neurons over the whole state space to find a linear approximation of Q at s=(x,v)
        # given the weights `w` of some action's features

        # 1.1) Map s=(x,v) to neurons based on the activation function
        # TODO: vectorize this (using meshgrid)
        s_prime = np.empty((self.side_size, self.side_size))
        for i in range(self.side_size):
            for j in range(self.side_size):
                s_prime[i,j] = self.activation(i,j, x,v)

        # 1.2) For each action `a_i`, compute Q(s, a_i) -- Vectorised
        # weights(num_actions x num_neurons_pos x num_neurons_v) . states(neuron_activ_pos x neuron_activ_v) = (num_Q_actions)
        # i.e. (3x20x20) . (20x20) -> (3,)
        # out_activations = np.tensordot(self.weights, s_prime, 2)
        Q_sp_a = np.tensordot(self.weights, s_prime, 2)

        # ε-greedy action choice; Generate action probability with `tau` exploration temperature
        action_probabilities = softmax(Q_sp_a, self.tau)
        a_prime = np.random.choice([-1,0,1], p=action_probabilities)

        return a_prime, s_prime, Q_sp_a[a_prime+1]

    def learn(self, state, action, Q_s_a, Q_sp_ap, eta = .01, gamma = .99, lambdaa = 0.90):
        """ Updates the weights of all actions based on the observed reward and a decaying eligibility trace """
        d = self.mountain_car.R - (Q_s_a - gamma*Q_sp_ap)

        # first decay the eligibility trace
        self.eligibility_trace *= gamma * lambdaa

        # then reinforce it based on the actions that caused it
        self.eligibility_trace[action+1] += state #action belongs to {-1,0,1}

        # finally, update the weights with this new one
        self.weights += eta * d * self.eligibility_trace

