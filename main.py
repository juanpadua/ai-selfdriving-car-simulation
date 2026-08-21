import pygame
import os
import pickle
import numpy as np
from navtraining import run_neat
from simulation import run_simulation
from environment import screen,map

os.chdir(os.path.dirname(__file__))

while True:
    pygame.init()
    choice=int(input('1.Train a model\n2.Run a simulation with a trained model\n'))
    if choice==1:
        choice2=int(input('1.Phase 1 ( Navigation Training )\n2.Phase 2 ( Car Detection Training )'))
        if choice2==1:
            n=1
            while os.path.exists(f'data/cars/models/model-{n}.bin'):
                n+=1
            f=open(f'data/cars/models/model-{n}.bin','wb')
            model=run_neat()
            pickle.dump(model,f)
            f.close()
    if choice==2:
        genomes=[]
        q=input('do you want to control a bot (y/n)')
        n=int(input('no of ai cars: '))
        while n>0:
            m=int(input('Enter model number :'))
            f=open(f'data/cars/models/model-{m}.bin','rb')
            model=pickle.load(f)
            genomes.append(model)
            n-=1
        if q=='y':
            run_simulation(genomes,True)
        else:
            run_simulation(genomes,False)
    else:
        pygame.quit()
        break