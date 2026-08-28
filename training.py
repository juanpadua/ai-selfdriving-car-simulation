import neat
import pygame
from environment import Car,screen,map,mouse_click
from settings import generations,frames_per_gen
import pickle
import os

os.chdir(os.path.dirname(__file__))

gen=generations
set_pos=False
x=0
y=0
n=1
def eval_genomes(genomes,config):
    global gen
    global set_pos
    global x
    global y
    global n
    gen-=1
    clock = pygame.time.Clock()
    cars=[]
    car_list=[]
    if not set_pos:
        x=0
        y=0
        pygame.display.set_caption('Place Your Car Genome Spawn Position :')
        x,y=mouse_click()
        set_pos=True
    for i,(id,genome) in enumerate(genomes):
        net=neat.nn.FeedForwardNetwork.create(genome,config)
        car=Car(x,y,net)
        cars.append((car,genome))
        car_list.append(car)
        genome.fitness=0
    running=True
    for f in range(frames_per_gen):
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
        if not running:
            break
        screen.blit(map,(0,0))
        all_dead=True
        for car,genome in cars:
            if car.alive:
                all_dead=False
                car.controls(car_list)
                car.math(training=True)
                car.draw()
                genome.fitness+=car.get_reward()
        clock.tick(60)
        if f%2==0:
            pygame.display.set_caption(f"Training ({frames_per_gen-f} frames left)\t\tGeneration : {generations-gen}")
            pygame.display.flip()
        if all_dead:
            break
    best_fitness=-100000
    for id,genome in genomes:
        if genome.fitness>best_fitness:
            best_genome=genome
            best_fitness=genome.fitness
    with open(f'data/cars/models/cache/gen-best-{n}','wb') as f:
        pickle.dump(best_genome,f)
    n+=1
def run_neat(configfile='config.txt'):
    global set_pos
    global n
    config=neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,configfile)
    population=neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    winner=population.run(eval_genomes, generations)
    set_pos=False
    n=1
    best_fitness=-100000
    for i in range(generations):
        with open(f'data/cars/models/cache/gen-best-{i+1}','rb') as f:
            genome=pickle.load(f)
            if genome.fitness>best_fitness:
                best_genome=genome
                best_fitness=genome.fitness
    return best_genome