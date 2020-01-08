'''
Name:        DQN Self autonomous robot with 6 outputs for NN (the 6 actions)

Author:      Thomas Specq
Website:     http://www.thomas-specq.work
Link:        <a href="http://www.thomas-specq.work">Freelance Web Design & Développement</a>
Created:     24/08/2017
Copyright:   (c) Thomas Specq 2017
Licence:     BSD
Video of the vehicules: https://www.youtube.com/watch?v=DdRTxZAj2eI
Requirements:

RaspberryPi
Robot Chassis
4 wheel DC motors
L298N Dual H Bridge
5 sonars HC SR04 
Power bank for the RPI
8 Battery AA
L3G4200D Gyrometer
Your valuable time

'''

import numpy as np
import time
import smbus
import json
import string
import RPi.GPIO as GPIO
import random
from random import randint

GPIO.setwarnings(False)

#Global Variables being established
# Q matrix
Q = np.matrix(np.zeros([6,6]))
#Flag are used to record the presence of obstacles
flag = [0,0,0,0,0,0];
flags = [0,0,0,0,0,0];
#available action at the moment
available_act = 0
#action taken
action = 0
#neural network layer
l1 = 0
l2 = 0
l3 = 0
#X = input of the NN
#Xs = past input of the NN
#Y = output of the NN
X = np.array([[0,0,0,0,0,0]])
Xs = np.array([[0,0,0,0,0,0]])               
Y = np.array([[0,0,0,0,0,0]]) 

state = 0

next_step_index = 0
#Count of the loop
loopCount = 1
#Motor Class for a PI L298N Dual H Bridge DC Motor driver
# 6 moving actions are created and 1 to stop
# a gyrometer will record the acceleration during the movement
# with the function acc.getAcc()
class Motor:
	global Y
	
	def __init__(self):
		self.Pin1 = 7
		self.Pin2 = 11
		self.Pin3 = 15
		self.Pin4 = 13
		GPIO.setmode(GPIO.BOARD)
		GPIO.setup(self.Pin1, GPIO.OUT)
		GPIO.setup(self.Pin2, GPIO.OUT)
		GPIO.setup(self.Pin3, GPIO.OUT)
		GPIO.setup(self.Pin4, GPIO.OUT)

	def goUp(self):
		acc = Accelerometer()
		print "Going Up"
		GPIO.output(self.Pin3, GPIO.HIGH)
		GPIO.output(self.Pin1, GPIO.HIGH)
		time.sleep(0.35)
		acc.getAcc()
		time.sleep(0.15)
		GPIO.output(self.Pin3, GPIO.LOW)
		GPIO.output(self.Pin1, GPIO.LOW)
		
	def goLeftB(self):
		acc = Accelerometer()
		print "Going Left a Bit"
		GPIO.output(self.Pin1, GPIO.HIGH)
		GPIO.output(self.Pin4, GPIO.HIGH)
		time.sleep(0.05)
		acc.getAcc()
		time.sleep(0.15)
		GPIO.output(self.Pin1, GPIO.LOW)
		GPIO.output(self.Pin4, GPIO.LOW)

	def goLeft(self):
		acc = Accelerometer()
		print "Going Left"
		GPIO.output(self.Pin1, GPIO.HIGH)
		GPIO.output(self.Pin4, GPIO.HIGH)
		time.sleep(0.07)
		acc.getAcc()
		time.sleep(0.17)
		GPIO.output(self.Pin1, GPIO.LOW)
		GPIO.output(self.Pin4, GPIO.LOW)

	def goDown(self):
		acc = Accelerometer()
		print "Going Down"
		GPIO.output(self.Pin4, GPIO.HIGH)
		GPIO.output(self.Pin2, GPIO.HIGH)
		time.sleep(0.20)
		acc.getAcc()
		time.sleep(0.40)
		GPIO.output(self.Pin4, GPIO.LOW)
		GPIO.output(self.Pin2, GPIO.LOW)
		
	def goRightB(self):
		acc = Accelerometer()
		print "Going Right a Bit"
		GPIO.output(self.Pin3, GPIO.HIGH)
		GPIO.output(self.Pin2, GPIO.HIGH)
		time.sleep(0.10)
		acc.getAcc()
		time.sleep(0.15)
		GPIO.output(self.Pin3, GPIO.LOW)
		GPIO.output(self.Pin2, GPIO.LOW)

	def goRight(self):
		acc = Accelerometer()
		print "Going Right"
		GPIO.output(self.Pin3, GPIO.HIGH)
		GPIO.output(self.Pin2, GPIO.HIGH)
		time.sleep(0.07)
		acc.getAcc()
		time.sleep(0.17)
		GPIO.output(self.Pin3, GPIO.LOW)
		GPIO.output(self.Pin2, GPIO.LOW)

	def stop(self):
		print "Stopping"
		GPIO.output(self.Pin3, GPIO.LOW)
		GPIO.output(self.Pin2, GPIO.LOW)
		GPIO.output(self.Pin1, GPIO.LOW)
		GPIO.output(self.Pin4, GPIO.LOW)	

	def perform(self, action):
		if action == 0:
			self.goUp()
		elif action == 1:
			self.goRight()
		elif action == 2:
			self.goLeft()
		elif action == 3:
			self.goRightB()
		elif action == 4:
			self.goLeftB()
		elif action == 5:
			self.goDown()

		time.sleep(0.20)
		
#The ai class is a basic Neural Network with sigmoid function
#We will use it to predict the outcome of our Q learning next step
class AI:
	def nonlin(self,x,deriv=False):
		if(deriv==True):
		    return x*(1-x)
		return 1/(1+np.exp(-x))

	def NNTrain(self, X, Xs, Y, a):
		global syn0
		global syn1
		global syn2
		global l1
		global l2
		global l3

		l0 = X
		l1 = self.nonlin(np.dot(l0,syn0))
		l2 = self.nonlin(np.dot(l1,syn1))
    		l3 = self.nonlin(np.dot(l2,syn2))

		

   		l3_error = Y - l3
    		l3_delta = l3_error*self.nonlin(l3,deriv=True)
    		l2_error = l3_delta.dot(syn2.T)
		l2_delta = l2_error*self.nonlin(l2,deriv=True)
		l1_error = l2_delta.dot(syn1.T)
		l1_delta = l1_error * self.nonlin(l1,deriv=True)

    		syn2 += l2.T.dot(l3_delta)
    		syn1 += l1.T.dot(l2_delta)
   		syn0 += l0.T.dot(l1_delta)
   		print "L3 ",l3
   		return l3
   		
	def NNTest(self, X):
	
		l0 = X
		l1 = self.nonlin(np.dot(l0,syn0))
		l2 = self.nonlin(np.dot(l1,syn1))
    		l3 = self.nonlin(np.dot(l2,syn2))
    		
   		print "L3 NN test for Qlearning ",l3
   		return l3 
	#This function is not mandatory to use
	#We can pre train the NN to act as wanted 
	#For example got to the right when an obstacle on the left is found
	#But a true AI should be ableto learn by itselfand this method wont be called
	def trainSchedule(self):
		X = np.array([[1,0,0,0,0,0],
				[1,0,0,0,0,0],
				[1,0,0,0,0,0],
				[1,0,0,0,0,0],
				[1,0,0,0,0,0],
				[0,1,0,0,0,0]
				]) 

		Y = np.array([[1,0,0,0,0,0],
				[0,0,0,1,0,0],
				[0,0,0,0,1,0],
				[0,0,1,0,0,0],
				[0,1,0,0,0,0],
				[0,0,0,0,0,1]
				])

		for a in range(0,500):
			self.NNTrain(X, Y)
#The Q learning class that will reward or punish the robot if a good or bad action is taken
class Qlearning:
   	def __init__(self):
            self.gamma = 0.8
            self.data = {}
	    self.data['inputOutput'] = []
            
	    # R matrix
	    self.R = np.matrix([ [0,0,0,0,0,0],
		[0,0,0,0,0,0],
		[0,0,0,0,0,0],
		[0,0,0,0,0,0],
		[0,0,0,0,0,0],
		[0,0,0,0,0,0] ])

	# This function returns all available actions in the state given as an argument
	def available_actions(self, state):
	    print "state ",state
	    current_state_row = self.R[state,]
	    av_act = np.where(current_state_row >= 0)[1]
	    return av_act
	    
	def setNewState(self):
	    global state
	    global action
	    global X  
	    global flag
	    sonar = Sonar()
	    motor = Motor()
	    
	    sonar.checkDistanceAll()
	    	
	    #Test if wall presence and take action to go back and more right a bit if its the case
	    print "flags",sum(flags)
	    print "flag",sum(flag)
	    if (sum(flag) > 0 and sum(flags) > 0):  
		    #Punish if the robot touched a wall
		    print "PUNITION"
		    print "action test",action
		    print "current state test",state    
		    Q[state, action] = -20
		    #cheat code
		    #if(state == 1 and action == 5):
		    # 		Q[state, action] += 100
		    		 	    		 
  	     
	# This function chooses at random which action to be performed within the range 
	# of all the available actions.
	def sample_next_action(self, available_actions_range):
	    global action
	    global X
	    global next_step_index
	    global loopCount
	    
	    motor = Motor()
	    available_act1 = available_actions_range
	    print "LOOPCOUNT",loopCount
	    
	    #Epsilon formula
	    epsilon = 100-(np.log(loopCount)*1.5/14)*100
	    epsilon = np.around(epsilon/100,2)
	    print "Epsilon : ",epsilon
	
	    #Cheat code here; the random number are generated between 0 and 0.8, so we have more random phase at the start
	    random1 = random.uniform(0,0.8)
	    random1 = np.around(random1,2)
	    print "Random float : ",random1	    
	    
	    if(random1 <= epsilon):
	    	action = int(np.random.choice(available_act1,1))
	    	print "RANDOM"
	    	
	    else:
		next_step_index = np.where(Q[state,] == np.max(Q[state,]))[1]
		if next_step_index.shape[0] > 1:
			print "SELECTION : MULTIPLE"
			next_step_index = int(np.random.choice(next_step_index, size = 1))
			action = int(next_step_index)
		else:
			print "SELECTION : 1"
			action = int(next_step_index)

	    print "action to be done",action	  
	    
	    motor.perform(action)
	    
	    
	    
	    print "action ",action
	    if(action == 0):
	    	Y.flat[0] = 1
	    elif(action == 1):
		Y.flat[1] = 1
	    elif(action == 2):
		Y.flat[2] = 1
	    elif(action == 3):
		Y.flat[3] = 1
	    elif(action == 4):
		Y.flat[4] = 1
	    elif(action == 5):
		Y.flat[5] = 1
	    return action
	
	#stock X Y and loop count in memory, use it to train at random the NN
	def replayMemory(self):
	    global data	
    	    ai = AI()
    	    
    	    self.data['inputOutput'].append({'Iteration': loopCount,'X': X.tolist(),'Y': Y.tolist(),'Xs': Xs.tolist(),'Action': action})
	    
	    with open('dqn.json', 'w') as fp:
    		    json.dump(self.data, fp)
	    
	    # Use the NN to find the value of next action upon this state and action taken    
	    with open('dqn.json') as json_file:  
		    data = json.load(json_file)
		    randomNum = random.randint(0,len(self.data['inputOutput']))-1
		    
		    print randomNum
		    print "random number ",randomNum
		    
		    iteration = (self.data['inputOutput'][randomNum]['Iteration'])
		    print "Iteration ",iteration
		    goodX = (self.data['inputOutput'][randomNum]['X'])
		    goodX = np.array(goodX)
		    print "X for train ",goodX
		    goodXs = (self.data['inputOutput'][randomNum]['Xs'])
		    goodXs = np.array(goodXs)
		    print "Xs for train ",goodXs
		    goodY = (self.data['inputOutput'][randomNum]['Y'])
		    goodY = np.array(goodY)
		    print "Y for train ",goodY
		    pastAction = (self.data['inputOutput'][randomNum]['Action'])	    
		    print "Next action ",nextAction
	    	    ai.NNTrain(goodX, goodXs, goodY, pastAction)    	    	   

	# This function updates the Q matrix according to the path selected and the Q 
	# learning algorithm
	def update(self, current_state, action, gamma): 
	    global Y
	    global X
	    ai = AI()
	    
	    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
	    if max_index.shape[0] > 1:
		max_index = int(np.random.choice(max_index, size = 1))
	    else:
		max_index = int(max_index)
	    	    
	    max_value = round(ai.NNTest(X),2)
	    
	    #Max Value replaced by output NN
	    print "good X",X
	    print "good Y",Y
	    print "MAX VALUE ",max_value
	    #modified for the robot purpose, put += so it cumulate or just = so it get modified everytime
	    Q[current_state, action] = max_value * 100
	    print "action",action
	    print "current state",current_state	    
	    print "Qcurrent state",Q[current_state, action]
	      
	def all(self, current_state):
		global X
		global Y
		available_act = self.available_actions(current_state)
		action = self.sample_next_action(available_act)
		self.update(current_state,action,self.gamma)
		#train the NN from past data collected
	    	self.replayMemory()	
	    	    		 
		X = np.array([[0,0,0,0,0,0]])
		Xs = np.array([[0,0,0,0,0,0]]) 
		Y = np.array([[0,0,0,0,0,0]])  
		self.setNewState()
#Sonar class that evaluate the distance from the vehicules to the walls
#5 sonars HC SR04 are trigerred in orderly fashion
class Sonar:	
	def __init__(self):
		self.TRIG1 = 18
		self.ECHO1 = 16
		self.TRIG2 = 32
		self.ECHO2 = 22
		self.TRIG3 = 38
		self.ECHO3 = 36
		self.TRIG4 = 31
		self.ECHO4 = 29
		self.TRIG5 = 35
		self.ECHO5 = 33
		self.distance1 = 0
		self.distance2 = 0
		self.distance3 = 0
		self.distance4 = 0
		self.distance5 = 0
		GPIO.setmode(GPIO.BOARD)
		GPIO.setup(self.TRIG1,GPIO.OUT)
		GPIO.setup(self.ECHO1,GPIO.IN)
		GPIO.setup(self.TRIG2,GPIO.OUT)
		GPIO.setup(self.ECHO2,GPIO.IN)
		GPIO.setup(self.TRIG3,GPIO.OUT)
		GPIO.setup(self.ECHO3,GPIO.IN)
		GPIO.setup(self.TRIG4,GPIO.OUT)
		GPIO.setup(self.ECHO4,GPIO.IN)
		GPIO.setup(self.TRIG5,GPIO.OUT)
		GPIO.setup(self.ECHO5,GPIO.IN)

	def clean1(self):
		GPIO.output(self.TRIG1, False)
		time.sleep(0.0001)

	def clean2(self):
		GPIO.output(self.TRIG2, False)
		time.sleep(0.0001)

	def clean3(self):
		GPIO.output(self.TRIG3, False)
		time.sleep(0.0001)

	def clean4(self):
		GPIO.output(self.TRIG4, False)
		time.sleep(0.0001)

	def clean5(self):
		GPIO.output(self.TRIG5, False)
		time.sleep(0.0001)
	
	def cleanAll(self):
		self.clean1()
		self.clean2()
		self.clean3()
		self.clean4()
		self.clean5()

	def fire1(self):
		self.clean1()
		GPIO.output(self.TRIG1, True)
		time.sleep(0.00001)
		GPIO.output(self.TRIG1, False)

		while GPIO.input(self.ECHO1)==0:
			pulse_start = time.time()

		while GPIO.input(self.ECHO1)==1:
			pulse_end = time.time() 

		pulse_duration = pulse_end - pulse_start
		self.distance1 = pulse_duration*17150
		self.distance1 = round(self.distance1, 2)
		print "Sonar 1: ",self.distance1,"cm"  

	def fire2(self):
		self.clean2()
		GPIO.output(self.TRIG2, True)
		time.sleep(0.00001)
		GPIO.output(self.TRIG2, False)

		while GPIO.input(self.ECHO2)==0:
			pulse_start = time.time()

		while GPIO.input(self.ECHO2)==1:
			pulse_end = time.time() 

		pulse_duration = pulse_end - pulse_start
		self.distance2 = pulse_duration*17150
		self.distance2 = round(self.distance2, 2)
		print "Sonar 2: ",self.distance2,"cm" 

	def fire3(self):
		self.clean3()
		GPIO.output(self.TRIG3, True)
		time.sleep(0.00001)
		GPIO.output(self.TRIG3, False)

		while GPIO.input(self.ECHO3)==0:
			pulse_start = time.time()

		while GPIO.input(self.ECHO3)==1:
			pulse_end = time.time() 

		pulse_duration = pulse_end - pulse_start
		self.distance3 = pulse_duration*17150
		self.distance3 = round(self.distance3, 2)
		print "Sonar 3: ",self.distance3,"cm" 

	def fire4(self):
		self.clean4()
		GPIO.output(self.TRIG4, True)
		time.sleep(0.00001)
		GPIO.output(self.TRIG4, False)

		while GPIO.input(self.ECHO4)==0:
			pulse_start = time.time()

		while GPIO.input(self.ECHO4)==1:
			pulse_end = time.time() 

		pulse_duration = pulse_end - pulse_start
		self.distance4 = pulse_duration*17150
		self.distance4 = round(self.distance4, 2)
		print "Sonar 4: ",self.distance4,"cm"

	def fire5(self):
		self.clean5()
		GPIO.output(self.TRIG5, True)
		time.sleep(0.00001)
		GPIO.output(self.TRIG5, False)
		#Copyright Specq Thomas

		while GPIO.input(self.ECHO5)==0:
			pulse_start = time.time()

		while GPIO.input(self.ECHO5)==1:
			pulse_end = time.time() 

		pulse_duration = pulse_end - pulse_start
		self.distance5 = pulse_duration*17150
		self.distance5 = round(self.distance5, 2)
		print "Sonar 5: ",self.distance5,"cm"

	#Distance in order to rise a flag set initially at 25cm changed to 35cm to avoid collisions
	def checkDistance(self):
		global flag
		global flags
		global X
		global Xs
		
		
		flags = flag
		X = flag
		#flag set to 0 in case, possible repetition
		flag = [0,0,0,0,0]
		
		if self.distance1 < 35:      
			flag[0] = 1
		if self.distance2 < 35:      
			flag[1] = 1
		if self.distance3 < 35:      
			flag[2] = 1
		if self.distance4 < 35:      
			flag[3] = 1
		if self.distance5 < 35:      
			flag[4] = 1	
						
		Xs = flag		
		
		#Main sonar class function
	def checkDistanceAll(self):
		global X
		self.clean1()
		self.fire1()
		self.clean2()
		self.fire2()
		self.clean3()
		self.fire3()
		self.clean4()
		self.fire4()
		self.clean5()
		self.fire5()
		self.checkDistance()
#Gyrometer class used to evaluate the robot acceleration
#Trigerred upon a movement action is taken
#A positive acceleration will constitue a reward for the AI
class Accelerometer:

	#converts 16 bit two's compliment reading to signed int
	def getSignedNumber(self,number):
	    if number & (1 << 15):
		return number | ~65535
	    else:
		return number & 65535
		
	def getAcc(self):
		global Y
		#open /dev/i2c-1
		i2c_bus=smbus.SMBus(1)
		#i2c slave address of the L3G4200D
		i2c_address=0x69
		#normal mode and all axes on to control reg1
		i2c_bus.write_byte_data(i2c_address,0x20,0x0F)
		#full 2000dps to control reg4
		i2c_bus.write_byte_data(i2c_address,0x23,0x20)
		
		xDelta = 0
		yDelta = 0
		zDelta = 0
	
		for cycle in range (0,20):
			i2c_bus.write_byte(i2c_address,0x28)
			X_L = i2c_bus.read_byte(i2c_address)
			i2c_bus.write_byte(i2c_address,0x29)
			X_H = i2c_bus.read_byte(i2c_address)
			X = X_H << 8 | X_L

			i2c_bus.write_byte(i2c_address,0x2A)
			Y_L = i2c_bus.read_byte(i2c_address)
			i2c_bus.write_byte(i2c_address,0x2B)
			Y_H = i2c_bus.read_byte(i2c_address)
			Y0 = Y_H << 8 | Y_L

			i2c_bus.write_byte(i2c_address,0x2C)
			Z_L = i2c_bus.read_byte(i2c_address)
			i2c_bus.write_byte(i2c_address,0x2D)
			Z_H = i2c_bus.read_byte(i2c_address)
			Z = Z_H << 8 | Z_L

			xDelta = self.getSignedNumber(X)
			yDelta = self.getSignedNumber(Y0)
			zDelta = self.getSignedNumber(Z)
		           
		        # Cheat code due to failed gyrometer, only going fw action will be rewarded   
		        # remove is gyro pb solved      
		        if(abs(xDelta) >= 600):
                        	if(abs(zDelta) <= 1600):
                                	if(abs(yDelta) <= 800):
                                		if(action==0):
		                        		print "xDelta = ", xDelta
		                        		Y = 1
		                        		print "GOING FW"
		                        		break
                	else:
                        	if(abs(zDelta) >= 1300):
                                        print "GOING RIGHT OR LEFT "
 					Y = 0.4
                                        break
                        	else:
                        		Y = 0
				
		print "Y to be reported",Y
		

#-------------------------------------------------------------------------------
# Finally our main class
def main():
	global syn0
	global syn1
	global syn2
	global X
	global Y
	global state
	global action
	global loopCount
	
	time.sleep(1)
	print "Starting"
	
	qlear = Qlearning()
	ai = AI()
	#Random seed chosen for reproductibility
        np.random.seed(3090)
	np.set_printoptions(precision=0)
        syn0 = 2*np.random.random((6,12)) - 1 
        syn1 = 2*np.random.random((12,12)) - 1
        syn2 = 2*np.random.random((12,6)) - 1	
	
	#training the NN	
	#ai.trainSchedule()	
	#For loop to perform the actions
	#Asked to perform it 100 times in my case
	for loopCount in range(1,100):
		qlear.all(state)  
		# Normalize the "trained" Q matrix
		print "loop :",loopCount
		print("Trained Q matrix:")
		print(Q)
		print

	print "Ending"

if __name__ == '__main__':
    main()

