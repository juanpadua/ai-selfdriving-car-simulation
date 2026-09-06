import pygame
import os
import pickle
import numpy as np
from training import run_neat
from simulation import run_simulation

os.chdir(os.path.dirname(__file__))

while True:
    pygame.init()
    choice=int(input('1.Train a model\n2.Run a simulation with a trained model\n'))
    if choice==1:
        n=1
        while os.path.exists(f'data/cars/models/model-{n}.bin'):
            n+=1
        f=open(f'data/cars/models/model-{n}.bin','wb')
        model=run_neat()
        pickle.dump(model,f)
        f.close()
    if choice==2:
        genomes=[]
        q2=input('do you want to control a bot (y/n)')
        n=int(input('no of ai cars: '))
        while n>0:
            m=int(input('Enter model number :'))
            f=open(f'data/cars/models/model-{m}.bin','rb')
            model=pickle.load(f)
            genomes.append(model)
            n-=1
        if q2=='y':
            run_simulation(genomes,True)
        else:
            run_simulation(genomes,False)
    else:
        pygame.quit()
        break