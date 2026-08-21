import pygame
import time
import math
import numpy as np
import os
from settings import scale_x

os.chdir(os.path.dirname(__file__))
pygame.init()

#MAP ASSETS/CONSTANTS
mapx=scale_x
mapy=scale_x*9//16
screen=pygame.display.set_mode((mapx,mapy))
map=pygame.image.load('data/tracks/track-4.png').convert()
map=pygame.transform.scale(map,(mapx,mapy))
map_array=pygame.surfarray.array3d(map)
bad_color=(255,255,255)
wall_mask=np.all(map_array==bad_color,axis=2)

#CAR ASSETS/CONSTANTS
car_w=int((scale_x*75//960)*0.6)
car_h=int((scale_x*62//960)*0.6)
speed_cap=2*scale_x//960
car_sprite_1=pygame.image.load(f'data/cars/car1.png').convert_alpha()
car_sprite_1=pygame.transform.scale(car_sprite_1,(car_w,car_h))

class Car:
    def __init__(self,x,y,net):
        self.car=car_sprite_1.copy()
        self.rotated_car=self.car
        self.car_hitbox=pygame.Rect(x,y,car_w,car_h)
        self.velocity=0
        self.friction=0.95
        self.angle=0
        self.rotation=0
        self.x=float(x)
        self.y=float(y)

        self.net=net
        self.alive=True
        self.raylen=80
        self.sim_time=time.time()
        self.elapsed_idle_time=0
        self.spawn_time=time.time()
        self.half_len=int(car_w//2)
        self.half_len_h=int(car_h//2)
        self.death_threshold=1-self.half_len/self.raylen

    def get_sensors(self,cars):
        self.center_x=self.car_hitbox.centerx
        self.center_y=self.car_hitbox.centery
        self.rayangles=[-30,0,30,180]
        self.sensors=[0,0,0,0]
        self.car_sense=[0,0,0,0]

        nearby=[]
        for c in cars:
            if c is self:
                continue
            x=abs(c.car_hitbox.centerx - self.center_x)
            y=abs(c.car_hitbox.centery - self.center_y)
            if x<self.raylen and y<self.raylen:
                nearby.append(c)

        for value, self.rayangle in enumerate(self.rayangles):
            self.rayanglerad=math.radians(self.angle + self.rayangle)
            car_collide=False
            
            for raystep in range(1,self.raylen+1):
                rayx=int(self.center_x + raystep * math.cos(self.rayanglerad))
                rayy=int(self.center_y + raystep * math.sin(self.rayanglerad))

                if not(0<rayx<mapx-1 and 0<rayy<mapy-1):
                    self.sensors[value]=1 - raystep/self.raylen
                    break

                if wall_mask[rayx, rayy]:
                    self.sensors[value]=1 - raystep/self.raylen
                    break

                for car in nearby:
                    if car is self:
                        continue
                    if car.car_hitbox.collidepoint(rayx,rayy):
                        self.car_sense[value]=1 - raystep/self.raylen
                        car_collide=True
                        break

                if car_collide:
                    break

            else:
                self.sensors[value]=0

        return self.sensors
    
    def user_control(self,keys):
        if keys[pygame.K_w]:
            self.velocity+=0.2*self.friction
        if keys[pygame.K_s]:
            self.velocity-=0.2*self.friction
        if keys[pygame.K_d]:
            self.angle+=self.rotation
        if keys[pygame.K_a]:
            self.angle-=self.rotation

        self.velocity = max(min(self.velocity, speed_cap), -speed_cap)

        if -0.2 < self.velocity < 0.2:
            self.rotation = 0
        else:
            self.rotation = 2

    def user_math(self):
        angle_radian=math.radians(self.angle)

        self.velocity*=self.friction

        self.x+=self.velocity*math.cos(angle_radian)
        self.y+=self.velocity*math.sin(angle_radian)

        self.car_hitbox.x =int(self.x)
        self.car_hitbox.y =int(self.y)

    def controls(self,cars):
        sensors = self.get_sensors(cars)
        output = self.net.activate(sensors)
        
        self.velocity += 0.2 * output[0] * self.friction

        self.velocity = max(min(self.velocity, speed_cap), -speed_cap)

        if -0.2 < self.velocity < 0.2:
            self.rotation = 0
        else:
            self.rotation = 2
        self.angle += self.rotation * output[1]

    def math(self):
        angle_radian=math.radians(self.angle)

        self.velocity*=self.friction

        self.x+=self.velocity*math.cos(angle_radian)
        self.y+=self.velocity*math.sin(angle_radian)

        self.car_hitbox.x =int(self.x)
        self.car_hitbox.y =int(self.y)

        if self.velocity > 0.1:
            self.elapsed_idle_time = 0
            self.sim_time = time.time()
        else:
            self.elapsed_idle_time = time.time() - self.sim_time

    def draw(self):
        self.car=pygame.transform.rotate(self.rotated_car,-self.angle)
        self.car_rect=self.car.get_rect(center=self.car_hitbox.center)
        screen.blit(self.car, self.car_rect)

    def get_reward(self):
        reward=0
        sensors=self.sensors
        reward+=self.velocity
        reward-=sensors[0]*10
        reward-=sensors[2]*10

        if sensors[1]>0.4:
            if self.velocity>=0:
                reward-=sensors[1]*10
            else:
                reward+=(1-sensors[1])*10
                reward+=abs(self.rotation)*3
        if sensors[3]>0.4:
            if self.velocity<=0:
                reward-=sensors[3]*10
            else:
                reward+=(1-sensors[3])*10
                reward+=abs(self.rotation)*3
        return reward
    
    def get_reward_phase2(self):
        reward=0
        sensors=self.sensors
        reward+=self.velocity
        reward-=sensors[0]*10
        reward-=sensors[2]*10
        reward-=self.car_sense[0]*15
        reward-=self.car_sense[2]*15

        if sensors[1]>0.4:
            if self.velocity>=0:
                reward-=sensors[1]*10
            else:
                reward+=(1-sensors[1])*10
                reward+=abs(self.rotation)*3
        if sensors[3]>0.4:
            if self.velocity<=0:
                reward-=sensors[3]*10
            else:
                reward+=(1-sensors[3])*10
                reward+=abs(self.rotation)*3
        if self.car_sense[1]>0.4:
            if self.velocity>=0:
                reward-=self.car_sense[1]*15
            else:
                reward+=(1-self.car_sense[1])*15
                reward+=abs(self.rotation)*6
        if self.car_sense[3]>0.4:
            if self.velocity<=0:
                reward-=self.car_sense[3]*15
            else:
                reward+=(1-self.car_sense[3])*15
                reward+=abs(self.rotation)*3
        return reward

#OTHER FUNCTIONS
def mouse_click():
    clicked=False
    x=0
    y=0
    while not clicked:
        screen.blit(map,(0,0))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type==pygame.MOUSEBUTTONDOWN:
                x,y=pygame.mouse.get_pos()
                clicked=True
        pygame.draw.circle(screen,(64,64,64),(x,y),2)
    return x,y