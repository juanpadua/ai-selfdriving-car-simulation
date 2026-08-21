import neat
import pygame
import time
from environment import Car,screen,map,mouse_click
from functools import partial

set_pos=False
x=0
y=0

def eval_genomes(genomes,config):
    global set_pos
    global x
    global y
    cars=[]
    car_list=[]
    if not set_pos:
        x=0
        y=0
        x,y=mouse_click()
        set_pos=True
    for i,(id,genome) in enumerate(genomes):
        net=neat.nn.FeedForwardNetwork.create(genome,config)
        car=Car(x,y,net)
        cars.append((car,genome))
        car_list.append(car)
        genome.fitness=0
    for f in range(2000):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        all_dead=True
        screen.blit(map,(0,0))
        for car,genome in cars:
            if car.alive:
                all_dead=False
                car.controls(car_list)
                car.math()
                car.draw()
                genome.fitness += car.get_reward()
                if max(car.sensors)>=car.death_threshold:
                    car.alive=False
                if car.elapsed_idle_time>2:
                    car.alive=False

        if f%2==0:
            pygame.display.flip()

        if all_dead:
            break

def run_neat(configfile='config.txt'):
    global set_pos
    config=neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,configfile)
    population=neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    winner=population.run(eval_genomes, 50)
    set_pos=False
    return winner