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

    def __init__(self, mountain_car = None, side_size = 10, tau = 0.05,
                 x_range = (-150, 30), v_range = (-15, 15), weights = None,
                 eta = 0.01, gamma = 0.95, lambdaa = 0.9):
        """ Makes a new agent with given parameters:
        Model:
            mountain_car : Instance of MountainCar
            side_size    : input neurons are arranged in a grid of this size -- scalar
            tau          : strategy exploration temperature -- scalar
            x_range      : range of positions to with input neurons -- 2-tuple
            v_range      : range of velocities to cover with input neurons -- 2-tuple
            weights      : from input neurons to output neurons -- array(3 x side_size x side_size)

        Learning:
            eta          : learning rate -- scalar << 1
            gamma        : future state discounting factor -- scalar (0.95 recommended)
            lambdaa      : eligibility decay rate -- scalar in (0,1)
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

        # neuron preference centres, widths:
        self.centres_x, self.sigma_x = np.linspace(x_range[0], x_range[1], side_size, endpoint=True, retstep=True)
        self.centres_v, self.sigma_v = np.linspace(v_range[0], v_range[1], side_size, endpoint=True, retstep=True)

        # we transpose one of the dimensions so that it will be broadcasted nicely with the other
        self.centres_x = np.atleast_2d(self.centres_x)
        self.centres_v = np.atleast_2d(self.centres_v).T

        # we always use sigma**2 in our calculations, so save that one instead
        self.sigma_x = self.sigma_x ** 2
        self.sigma_v = self.sigma_v ** 2

        # save the rest of the params
        self.tau   = tau
        self.eta   = eta
        self.gamma = gamma
        self.lambdaa = lambdaa
        self.side_size  = side_size

        # number of steps used per episode
        self.escape_times = []

    def episode(self, n_steps=2000, tau=0.05, animation=False, fig=None):
        """ Do an episode of maximum `n_steps`
            This also accepts the `tau` parameter, in case you want to update it
            Optionally, you can specify a figure where to draw this episode
        """

        # prepare for the visualization
        if animation:
            mv = mountaincar.MountainCarViewer(self.mountain_car)
            mv.create_figure(n_steps, n_steps, fig)

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
            if animation:
                mv.update_figure(n)

            # stop when goal was reached
            if self.mountain_car.R > 0.0:
                # print("reward obtained at t = ", self.mountain_car.t, end='\n\n')
                break

        # episode is finished, save number of steps
        self.escape_times.append(self.mountain_car.t)

    def visualize_field(self, fig=None):
        if not fig:
            fig = plb.figure()
        else:
            plb.figure(fig.number)
            fig.clf()

        actions = np.empty((self.side_size, self.side_size))
        state = np.empty_like(actions)
        cx = self.centres_x.flatten()
        cv = self.centres_v.flatten()

        # TODO: maybe super vectorise this, although for side_size=100 it would
        #       reach a tensor of 100**4

        # for each possible state
        for i in range(self.side_size):
            for j in range(self.side_size):
                state = self.activation(cx[i], cv[j])
                Q_s_a = np.tensordot(self.weights, state, 2)
                actions[i,j] = np.argmax(Q_s_a) - 1  # to match true action

        plb.quiver(cx, cv, actions, np.zeros_like(actions))

        # adjust axes to see the margins
        l, r, b, t = plb.axis()
        dx, dy = 0.1*(r - l), 0.1*(t - b)
        plb.axis([l - dx, r, b - dy, t + dy])
        plb.xlabel('Position (m)', fontsize=14)
        plb.ylabel(r'Speed ($\frac{m}{s}$)', fontsize=14)
        plb.title(r'''Force direction at episode {e}{n}($\lambda={l}, \tau={t}$)'''.format(
            e=self.mountain_car.t, n="\n", l=self.lambdaa, t=self.tau), fontsize=18)
        # return actions

    def activation(self, x, v):
        """ The activation of the neurons given (x, v) """
        # sigma is already squared !
        return np.exp(-(x-self.centres_x)**2 / (self.sigma_x)
                      -(v-self.centres_v)**2 / (self.sigma_v))


    def choose_action(self, x, v):
        """ Computes the action from state (x,v) following an ε-greedy strategy.
            It uses a linear approximation of the value function and softmax
            probability distribution.

            Return: (a, s, , Q[s,a])
        """

        # 1) find Q of each possible action from state s=(x,v)
        # This is approximated linearly from `s` and each action's 'feature vector'

        # We consider that each action is described by a set of features, whose weights are learnt
        # We use a tiling of neurons over the whole state space to find a linear approximation
        # of Q at s=(x,v) given the weights `w` of some action's features

        # 1.1) Map s=(x,v) to neurons activation
        s = self.activation(x,v)

        # 1.2) For each action `a_i`, compute Q(s, a_i) -- Vectorised
        # weights(num_actions x num_neurons_pos x num_neurons_v) . states(neuron_activ_pos x neuron_activ_v) = (num_Q_actions)
        # i.e. (3x20x20) . (20x20) -> (3,)
        Q_s_a = np.tensordot(self.weights, s, 2)

        # ε-greedy action choice; Generate action probability with `tau` exploration temperature
        action_probabilities = softmax(Q_s_a, self.tau)
        a = np.random.choice([-1,0,1], p=action_probabilities)

        return a, s, Q_s_a[a+1]

    def learn(self, state, action, Q_s_a, Q_sp_ap):
        """ Updates the weights of all actions based on the observed reward and a decaying eligibility trace """
        d = self.mountain_car.R - (Q_s_a - self.gamma * Q_sp_ap)

        # first, decay the eligibility trace
        self.eligibility_trace *= self.gamma * self.lambdaa

        # then, reinforce it based on the actions that caused it
        self.eligibility_trace[action+1] += state # action belongs to {-1,0,1}

        # finally, update the weights with this new one
        self.weights += self.eta * d * self.eligibility_trace

