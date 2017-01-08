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
            #print('\rt =', self.mountain_car.t)
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

    def __init__(self, mountain_car = None ,outputs_weights = None,input_side_size = 20, x_range = [-150, 30], xdot_range = [-15, 15]):
        
        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        if outputs_weights is None:
            self.outputs_weights = np.zeros((3,input_side_size*input_side_size))
            #self.outputs_weights = np.random.rand(3,input_side_size*input_side_size)
        else:
            self.outputs_weights = outputs_weights
            
        self.input_side_size = input_side_size
        self.x_range = x_range
        self.xdot_range = xdot_range
        
        self.eligibility_trace = np.zeros((3,input_side_size*input_side_size))
        
    def visualize_trial(self, n_steps = 200, tau = 1):
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
                action, Qs, input_activities = self.choose_action(x, xdot,tau)
            

            #car update
            self.mountain_car.apply_force(action)
            self.mountain_car.simulate_timesteps(100, 0.01)
            
            
            #s' state
            x2 = self.mountain_car.x
            xdot2 = self.mountain_car.x_d
            
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
                
                
    
    def choose_action(self, x, xdot,tau):
        # Generate activity for each input neuron
        input_activities = np.zeros(self.input_side_size*self.input_side_size)
        
        sigma = (self.x_range[1]-self.x_range[0])/(self.input_side_size-1)
        sigmadot = (self.xdot_range[1]-self.xdot_range[0])/(self.input_side_size-1)
        
        for i in range(self.input_side_size):
            for j in range(self.input_side_size):

                x0 = self.x_range[0]+j*sigma
                xdot0 = self.xdot_range[0]+i*sigmadot

                input_activities[i*self.input_side_size+j] = np.exp(-(x-x0)**2/(sigma)**2-(xdot-xdot0)**2/(sigmadot)**2)

        # Generate activity for each output neuron
        output_activities = np.zeros(3)
        for i in range(3):
            output_activities[i] = self.outputs_weights[i].dot(input_activities)


        # Generate action probability for each action
        action_probabilities = np.exp(output_activities/tau)/np.sum(np.exp(output_activities/tau))
        action = np.random.choice([-1,0,1],1,True,action_probabilities)
        
        return action, output_activities[action+1], input_activities
    
    def learn(self, action, Qs, Qss, input_activities, eta = 0.01, gamma = .95, lambdaa = 0.90):
        TD = self.mountain_car.R - (Qs - gamma*Qss)
        #print(TD)
        for i in range(self.input_side_size):
            for j in range(self.input_side_size):
                self.eligibility_trace[action+1,i*self.input_side_size+j] = gamma * lambdaa * self.eligibility_trace[action+1,i*self.input_side_size+j] + input_activities[i*self.input_side_size+j]
                
                self.outputs_weights[action+1,i*self.input_side_size+j] += eta * TD * self.eligibility_trace[action+1,i*self.input_side_size+j] #action belongs to {-1,0,1}
        
                

