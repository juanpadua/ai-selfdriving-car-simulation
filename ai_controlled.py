#MODULES
import pygame
import math
import neat
import time

mapx=960
mapy=540
screen=pygame.display.set_mode((mapx,mapy))

car_w=150//2
car_h=125//2

map=pygame.image.load('data/tracks/track2.png').convert()
map=pygame.transform.scale(map,(mapx,mapy))
bad_color=(255,255,255)


class Car:
    def __init__(self,image,x,y,net):
        self.car=pygame.image.load(f'data/cars/{image}').convert_alpha()
        self.car=pygame.transform.scale(self.car,(car_w,car_h))
        self.rotated_car=self.car
        self.car_hitbox=pygame.Rect(x,y,car_w,car_h)
        self.velocity=0
        self.acceleration=0.2
        self.friction=0.95
        self.angle=-90
        self.rotation=0

        self.net=net
        self.alive=True
        self.fitness=0
        self.raylen=80
        self.sim_time=time.time()
        self.elapsed_idle_time=0

    def get_sensors(self):
        self.center_x=self.car_hitbox.centerx
        self.center_y=self.car_hitbox.centery
        self.rayangles=[-30,0,30,180]
        self.sensors=[0,0,0,0]

        for value, self.rayangle in enumerate(self.rayangles):
            self.rayanglerad=math.radians(self.angle + self.rayangle)
            
            for raystep in range(1,self.raylen+1):
                rayx=int(self.center_x + raystep * math.sin(self.rayanglerad))
                rayy=int(self.center_y + raystep * math.cos(self.rayanglerad))

                if not(0<rayx<mapx-1 and 0<rayy<mapy-1):
                    self.sensors[value]=1 - raystep/self.raylen
                    break

                if map.get_at((rayx,rayy))==bad_color:
                    self.sensors[value]=1 - raystep/self.raylen
                    break
            else:
                self.sensors[value]=0

        return self.sensors

    def controls(self):
        sensors = self.get_sensors()
        output = self.net.activate(sensors)
            
        self.velocity+=self.acceleration * output[0]
        self.angle+=self.rotation * output[1]

        if output[0]<0:
            self.acceleration=0.1
        else:
            self.acceleration=0.2

        if self.velocity>1:
            self.rotation=3
        elif self.velocity<-1:
            self.rotation=-3
        else:
            self.rotation=0

    def math(self):
        angle_radian=math.radians(self.angle)

        self.velocity*=self.friction

        self.car_hitbox.x += self.velocity*math.sin(angle_radian)
        self.car_hitbox.y += self.velocity*math.cos(angle_radian)

    def draw(self):
        self.car=pygame.transform.rotate(self.rotated_car,self.angle-90)
        self.car_rect=self.car.get_rect(center=self.car_hitbox.center)
        screen.blit(self.car, self.car_rect)

    def draw_sensors(self):
        for rayangle, sensor_value in zip(self.rayangles, self.sensors):
            rayanglerad = math.radians(self.angle + rayangle)
            
            endx = int(self.center_x + self.raylen * math.sin(rayanglerad))
            endy = int(self.center_y + self.raylen * math.cos(rayanglerad))
            
            color = (int(255 * sensor_value), int(255 * (1 - sensor_value)), 0)
            
            pygame.draw.line(screen, color, (self.center_x, self.center_y), (endx, endy), 2)
            pygame.draw.circle(screen, color, (endx, endy), 4)

    def get_reward(self):
        reward=0
        left_sensor=self.sensors[0]
        front_sensor=self.sensors[1]
        right_sensor=self.sensors[2]
        back_sensor=self.sensors[3]

        if self.velocity>0.1:
            self.elapsed_idle_time=0
            self.sim_time=time.time()
            reward+=self.velocity*2
        else:
            self.elapsed_idle_time=time.time()-self.sim_time
            reward-=self.elapsed_idle_time*2

        if left_sensor>0.3 and self.angle<0:
            reward+=10
        elif left_sensor>0.3:
            reward-=10
        if right_sensor>0.3 and self.angle>0:
            reward+=10
        elif right_sensor>0.3:
            reward-=10
        if front_sensor>0.6 and self.velocity<0:
            reward+=15
        elif front_sensor>0.3:
            reward-=10
        if back_sensor>0.3 and self.velocity>0:
            reward+=10
        elif back_sensor>0.3:
            reward-=10

        return reward

def eval_genomes(genomes,config):
    cars=[]
    for id,genome in genomes:
        net=neat.nn.FeedForwardNetwork.create(genome,config)
        car=Car('car1.png',mapx//2,mapy-95,net)
        cars.append((car,genome))
        genome.fitness=0

    #training

    for f in range(1100):
        all_dead=True
        screen.blit(map,(0,0))
        for car,genome in cars:
            if car.alive:
                all_dead=False
                car.controls()
                car.math()
                car.draw()
                car.draw_sensors()
                car.fitness+=car.get_reward()
                genome.fitness=car.fitness

                if car.elapsed_idle_time>2:
                    car.alive=False

                if max(car.sensors)>=0.85:
                    car.alive=False

                if max(car.sensors)>=0 and abs(car.velocity)<0.1:
                    car.alive=False

        pygame.display.flip()

        if all_dead:
            break
        pygame.time.delay(10)
    
def run_neat(configfile='config.txt'):
    config=neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,configfile)
    population=neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    winner=population.run(eval_genomes, 50)
    return winner

pygame.init()

run_neat('config.txt')

pygame.quit()