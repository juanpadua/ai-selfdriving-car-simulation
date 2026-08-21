import neat
import os
import pygame
from environment import Car,screen,map,mouse_click

os.chdir(os.path.dirname(__file__))

def run_simulation(genomes,user=False,configfile='config.txt'):
    config=neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,configfile)
    cars=[]
    for genome in genomes:
        net=neat.nn.FeedForwardNetwork.create(genome,config)
        x,y=mouse_click()
        car=Car(x,y,net)
        cars.append(car)
        pygame.display.flip()
    if user:
        ux,uy=mouse_click()
        u_car=Car(ux,uy,None)
    running=True
    while running:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
        screen.blit(map,(0,0))
        if user:
            keys=pygame.key.get_pressed()
            u_car.user_control(keys)
            u_car.user_math()
            u_car.draw()
        for car in cars:
            if car.alive:
                if user:
                    car.controls(cars+[u_car])
                else:
                    car.controls(cars+[u_car])
                car.math()
                car.draw()
                if max(car.sensors)>=car.death_threshold:
                    car.velocity-=2
        if not user and not any(car.alive for car in cars):
            running = False
        pygame.display.flip()
        pygame.time.delay(10)