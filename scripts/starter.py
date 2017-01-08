import sys

import pylab as plb
import numpy as np
import mountaincar

class DummyAgent():
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, mountain_car = None, parameter1 = 3.0):
        
        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        self.parameter1 = parameter1

    def visualize_trial(self, n_steps = 200):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """
        
        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()
            
        # make sure the mountain-car is reset
        self.mountain_car.reset()

        for n in range(n_steps):
            print('\rt =', self.mountain_car.t)
            sys.stdout.flush()
            
            # choose a random action
            self.mountain_car.apply_force(np.random.randint(3) - 1)
            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            # update the visualization
            mv.update_figure()
            plb.draw()
            
            
            # check for rewards
            if self.mountain_car.R > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break

    def learn(self):
        # This is your job!
        pass
    
class Agent():
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, mountain_car = None, tau=1 ,outputs_weights = None,input_side_size = 20, x_range = [-150, 30], xdot_range = [-15, 15]):
        
        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        if outputs_weights is None:
            self.outputs_weights = np.ones((3,input_side_size*input_side_size))
            #self.outputs_weights = np.random.rand(3,input_side_size*input_side_size)
        else:
            self.outputs_weights = outputs_weights
            
        self.input_side_size = input_side_size
        self.x_range = x_range
        self.xdot_range = xdot_range
        self.tau = tau
    def visualize_trial(self, n_steps = 200):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """
        
        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()
            
        # make sure the mountain-car is reset
        #self.mountain_car.reset()
        
        Qs, Qss = 0,0
        action, input_activities = 0,[]
        
        for n in range(n_steps):
            #print('\rt =', self.mountain_car.t)
            sys.stdout.flush()
            
            # choose action
            x = self.mountain_car.x
            xdot = self.mountain_car.x_d
            
            if n==0:
                action, Qs, input_activities = self.choose_action(x, xdot, self.outputs_weights, self.tau, self.input_side_size, self.x_range, self.xdot_range)
            

            #car update
            self.mountain_car.apply_force(action)
            self.mountain_car.simulate_timesteps(100, 0.01)
            
            
            #s' state
            x2 = self.mountain_car.x
            xdot2 = self.mountain_car.x_d
            
            action2,Qs2,input_activities2 = self.choose_action(x2, xdot2, self.outputs_weights, self.tau, self.input_side_size, self.x_range, self.xdot_range)

            #learn
            self.outputs_weights = self.learn(self.outputs_weights,action, self.mountain_car.R,Qs, Qs2, input_activities, self.input_side_size)
            
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
                
    def choose_action(self, x, xdot, weights, tau, input_side_size, x_range, xdot_range):
        # Generate activity for each input neuron
        input_activities = np.zeros(input_side_size*input_side_size)
        
        sigma = (x_range[1]-x_range[0])/(input_side_size-1)
        sigmadot = (xdot_range[1]-xdot_range[0])/(input_side_size-1)
        
        for i in range(input_side_size):
            for j in range(input_side_size):

                x0 = x_range[0]+j*sigma
                xdot0 = xdot_range[0]+i*sigmadot

                input_activities[i*input_side_size+j] = np.exp(-(x-x0)**2/(sigma)**2-(xdot-xdot0)**2/(sigmadot)**2)

        # Generate activity for each output neuron
        output_activities = np.zeros(3)
        for i in range(3):
            output_activities[i] = weights[i].dot(input_activities)


        # Generate action probability for each action
        action_probabilities = np.exp(output_activities/tau)/np.sum(np.exp(output_activities/tau))
        action = np.random.choice([-1,0,1],1,True,action_probabilities)
        
        return action, output_activities[action+1], input_activities
    
    def learn(self, weights, action, R, Qs, Qss, input_activities, input_side_size, eta = 0.001, gamma = 0.95, lambdaa = 0.90):
        TD = R - (Qs - gamma*Qss)
        #print(TD)
        for i in range(input_side_size):
            for j in range(input_side_size):
                weights[action+1,i*input_side_size+j] += eta * TD * input_activities[i*input_side_size+j] #action belongs to {-1,0,1}
        
        return weights
                

