import time
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from queue import Queue

class Controller:
    S = list(range(-20, 150)) 
    A = [0, 1, 2]


    def __init__(self):
        self.n_episode = 100
        self.epsilon = 0.3
        self.alpha = 0.7
        self.gamma = 0.9

        self.accel = False
        self.brake = True

        self.curr_speed = 0
        self.curr_temperature = 10

        self.record = []
        
        self.Q = {(s, a): 0 for s in Controller.S for a in Controller.A}

        self.remain_distance = 100

        self.__optimal_temperature = 30

        self.max_speed = 70
        self.min_speed = 30

        self.reward = 0
        self.avg_temp = 0
        self.cnt = 0

    def accelerate(self):
        self.accel = True
        self.brake = False

    def not_accelerate(self):
        self.accel = False
        self.brake = False

    def braking(self):
        self.brake = True
        self.accel = False

    def epsilon_greedy(self, state):
        rand = random.random()
        
        if self.cnt <= 2 or self.curr_speed == 0:
            return Controller.A[0]
        
        elif rand < self.epsilon:
            return random.choice(Controller.A)
        
        else:
            q_values = [self.Q.get((state, a), 0) for a in Controller.A]
            max_q = max(q_values)
            return Controller.A[q_values.index(max_q)]


    def give_reward(self, state):
        diff = abs(self.__optimal_temperature - state)

        # reward for last ten records of temp
        if self.avg_temp > self.__optimal_temperature or self.avg_temp < self.__optimal_temperature:
            accumulated_reward = -10

        else:
            accumulated_reward = 0
        
        # reward for diff
        if diff == 0:
            reward_for_braking = 10

        else:
            reward_for_braking = 10 * 1/diff


        # reward for speed
        if self.curr_speed > self.max_speed:
            reward_for_speed = -self.curr_speed * 0.1
        
        elif self.curr_speed < self.min_speed:
            reward_for_speed = -self.curr_speed
        
        else:
            reward_for_speed = 5

        
        return reward_for_braking*1.5 + reward_for_speed + accumulated_reward*0.7
        
        

    def control(self, action):
        match action:
            case 0:
                self.accelerate()
            case 1:
                self.not_accelerate()
            case 2:
                self.braking()

        if self.accel:
            self.curr_speed += 7 
            self.curr_temperature -= 2 
        else:
            if self.brake:
                if self.curr_speed:
                    self.curr_temperature += 3
                    self.curr_speed -= 4 
                else:
                    if self.curr_speed <= 0:
                        self.curr_speed = 0.0
            else:
                self.curr_speed -= 1
                self.curr_temperature -= 1 

        self.curr_speed = min((max(0, self.curr_speed)), 100)
        self.curr_temperature = min((max(-20, self.curr_temperature)), 150)
        

    def drive(self):
        plt.ion()
        fig, ax = plt.subplots()
        
        x_data = []
        y_data = []
        speed_data = []
        total_speed_data = []
        brake_temp_avg = []
        
        avg_while_ten = Queue()
        size_of_queue = avg_while_ten.qsize()

        for i in range(self.n_episode):
            print("**************")
            print(f"Episode {i+1}")
            print("**************")
            self.remain_distance = 100
            self.accel = False
            self.brake = True

            self.curr_speed = 0
            self.curr_temperature = 10

            

            while self.remain_distance > 0:
        
                self.cnt += 1
                curr_state = self.curr_temperature
                avg_while_ten.put(curr_state)
                total_temp = 0
                avg_temp_list = []

                if avg_while_ten.qsize() > 10:
                    while not avg_while_ten.empty():
                        each = avg_while_ten.get()
                        total_temp += each
                        avg_temp_list.append(each)
                    avg_temp_list.pop(0)

                self.avg_temp = total_temp / 10
                print(f"avg_temp: {self.avg_temp}")

                for element in avg_temp_list:
                    avg_while_ten.put(element)

                x_data.append(self.cnt)
                y_data.append(curr_state)

                ax.clear()
                if self.curr_speed > self.max_speed or self.curr_speed < self.min_speed:
                    ax.plot(x_data, y_data, label='State over time', color='darkred')
                else:
                    ax.plot(x_data, y_data, label='State over time', color='blue')
                
                ax.set_xlabel('Time until arrival')
                ax.set_ylabel('Current Temperature of Brake')
                ax.set_title(f'Episode {i+1}, Target Temp: 30\'C')

                plt.draw()
                plt.pause(0.1)

                action = self.epsilon_greedy(curr_state)

                self.control(action)

                next_state = self.curr_temperature

                reward =  self.give_reward(next_state)

                old_q = self.Q[(curr_state, action)]

                self.Q[(curr_state, action)] += self.alpha * (
                    reward + self.gamma * max(self.Q.get((next_state, a), 0) for a in Controller.A) - old_q
                )

               
                if self.curr_speed > 80:
                    self.remain_distance -= self.curr_speed * 0.01
                else:
                    self.remain_distance -= 0.3
                speed_data.append(self.curr_speed)
                print(f"curr_temp: {curr_state}, action: {action}, next_temp: {next_state}, reward: {reward}")
                print(f"curr_speed: {self.curr_speed}, remain_distance: {self.remain_distance}")
                print("==================================")
            temp_avg = sum(y_data)/self.cnt
            speed_avg = sum(speed_data)/self.cnt
            print(f"avg_temp of brake: {temp_avg}")
            print(f"avg_speed: {speed_avg}")
            brake_temp_avg.append(temp_avg)
            total_speed_data.append(speed_avg)
            self.cnt = 0
            x_data = []
            y_data = []
            speed_data = []

        plt.ioff()
        plt.show()
        
        for i in range(len(brake_temp_avg)):
            print(f"Ep{i+1}: avg_temp {brake_temp_avg[i]}'C, avg_speed: {total_speed_data[i]}")


agent = Controller()
agent.drive()

