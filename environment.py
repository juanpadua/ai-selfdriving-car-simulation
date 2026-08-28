import pygame
import time
import math
import numpy as np
import os
from settings import scale_x,track

os.chdir(os.path.dirname(__file__))
pygame.init()

#MAP ASSETS/CONSTANTS
mapx=scale_x
mapy=scale_x*9//16
screen=pygame.display.set_mode((mapx,mapy))
map=pygame.image.load(f'data/tracks/{track}').convert()
map=pygame.transform.scale(map,(mapx,mapy))
map_array=pygame.surfarray.array3d(map)
bad_color=(255,255,255)
wall_mask=np.all(map_array==bad_color,axis=2)

#CAR ASSETS/CONSTANTS
car_w=int((scale_x*75//960)*0.6)
car_h=int((scale_x*62//960)*0.6)
speed_cap=2*scale_x//960
car_sprite=pygame.image.load(f'data/cars/car1.png').convert_alpha()
car_sprite=pygame.transform.scale(car_sprite,(car_w,car_h))

class Car:
    def __init__(self,x,y,net):
        self.car=car_sprite.copy()
        self.rotated_car=self.car
        self.car_hitbox=pygame.Rect(x,y,car_w,car_h)
        self.velocity=0
        self.friction=0.95
        self.angle=0
        self.rotation=0
        self.half_len=int(car_w//2)
        self.half_len_h=int(car_h//2)
        self.x=float(x)
        self.y=float(y)

        self.net=net
        self.raylen=120
        self.sim_time=time.time()
        self.elapsed_idle_time=0
        self.spawn_time=time.time()
        self.wall_hits=0
        self.death_threshold=1-self.half_len/self.raylen
        self.alive=True
        self.visited=set()
        self.death_pentalty=0

    def get_sensors(self,cars):
        rayangles=[-150,-60,-30,0,30,60,150,180]
        sensors=[0,0,0,0,0,0,0,0]
        car_sense=[0,0,0,0,0,0,0,0]

        nearby=[]
        for c in cars:
            if c is self:
                continue
            x=abs(c.car_hitbox.centerx - self.car_hitbox.centerx)
            y=abs(c.car_hitbox.centery - self.car_hitbox.centery)
            if x<self.raylen and y<self.raylen:
                nearby.append(c)

        for value, rayangle in enumerate(rayangles):
            rayanglerad=math.radians(self.angle + rayangle)
            car_collide=False
            
            for raystep in range(1,self.raylen+1):
                rayx=int(self.car_hitbox.centerx + raystep * math.cos(rayanglerad))
                rayy=int(self.car_hitbox.centery + raystep * math.sin(rayanglerad))

                if not(0<rayx<mapx-1 and 0<rayy<mapy-1):
                    sensors[value]=1 - raystep/self.raylen
                    break

                if wall_mask[rayx, rayy]:
                    sensors[value]=1 - raystep/self.raylen
                    break

                for car in nearby:
                    if car is self:
                        continue
                    if car.car_hitbox.collidepoint(rayx,rayy):
                        car_sense[value]=1 - raystep/self.raylen
                        car_collide=True
                        break

                if car_collide:
                    break

            else:
                sensors[value]=0

        return sensors,car_sense

    def user_math(self):
        angle_radian=math.radians(self.angle)

        self.velocity*=self.friction

        self.x+=self.velocity*math.cos(angle_radian)
        self.y+=self.velocity*math.sin(angle_radian)

        self.car_hitbox.x =int(self.x)
        self.car_hitbox.y =int(self.y)

    def math(self,training=False):
        angle_radian = math.radians(self.angle)
        self.velocity *= self.friction
        prev_x, prev_y = self.x, self.y
        self.x += self.velocity * math.cos(angle_radian)
        self.y += self.velocity * math.sin(angle_radian)
        self.car_hitbox.x = int(self.x)
        self.car_hitbox.y = int(self.y)

        if training:
            self.grid=(int(self.x)//20,int(self.y)//20)

            if self.velocity > 0.3:
                self.elapsed_idle_time = 0
                self.sim_time = time.time()
            else:
                self.elapsed_idle_time = time.time() - self.sim_time
            if self.elapsed_idle_time > 2:
                self.alive=False
        self.frontx = int(self.car_hitbox.centerx + self.half_len_h * math.cos(angle_radian))
        self.fronty = int(self.car_hitbox.centery + self.half_len_h * math.sin(angle_radian))
        self.backx  = int(self.car_hitbox.centerx - self.half_len_h * math.cos(angle_radian))
        self.backy  = int(self.car_hitbox.centery - self.half_len_h * math.sin(angle_radian))

        if 0 < self.frontx < mapx-1 and 0 < self.fronty < mapy-1:
            if wall_mask[self.frontx, self.fronty]:
                if training:
                    self.alive=False
                    self.death_pentalty=-1000
                else:
                    self.x, self.y = prev_x, prev_y
                    self.velocity = 0
        if 0 < self.backx < mapx-1 and 0 < self.backy < mapy-1:
            if wall_mask[self.backx, self.backy]:
                if training:
                    self.alive=False
                    self.death_pentalty=-1000
                else:
                    self.x, self.y = prev_x, prev_y
                    self.velocity = 0

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

    def controls(self,cars):
        sensors = self.get_sensors(cars)[0]

        self.front_back_diff = ((sensors[3]+sensors[4]+sensors[5]) - (sensors[0]+sensors[6]+sensors[7]))/3
        self.left_right_diff = ((sensors[1]+sensors[2]) - (sensors[4]+sensors[5]))/3

        output = self.net.activate([self.front_back_diff,self.left_right_diff])

        self.velocity += 0.2 * output[0] * self.friction
        self.velocity = max(min(self.velocity, speed_cap), -speed_cap)
        self.rotation = abs(self.velocity) / speed_cap * 3
        self.angle += self.rotation * output[1]
        self.last_steer = output[1]

    def s_controls(self,cars):
        wall_sense, car_sense = self.get_sensors(cars)

        combined_front = wall_sense[3]+wall_sense[4]+wall_sense[5]+car_sense[3]+car_sense[4]+car_sense[5]
        combined_back  = wall_sense[0]+wall_sense[6]+wall_sense[7]+car_sense[0]+car_sense[6]+car_sense[7]
        self.front_back_diff = (combined_front - combined_back) / 6

        combined_left  = wall_sense[1]+wall_sense[2] + car_sense[1]+car_sense[2]
        combined_right = wall_sense[4]+wall_sense[5] + car_sense[4]+car_sense[5]
        self.left_right_diff = (combined_left - combined_right) / 4

        output = self.net.activate([self.front_back_diff,self.left_right_diff])
        
        self.velocity += 0.2 * output[0] * self.friction
        self.velocity = max(min(self.velocity, speed_cap), -speed_cap)
        self.rotation = abs(self.velocity) / speed_cap * 3
        self.angle += self.rotation * output[1]
        self.last_steer = output[1]

    def adas(self,cars):
        car_sense=self.get_sensors(cars)[1]
        if max(car_sense[2],car_sense[3],car_sense[4])>0.8:
            self.velocity*=0.5

    def draw(self):
        self.car=pygame.transform.rotate(self.rotated_car,-self.angle)
        self.car_rect=self.car.get_rect(center=self.car_hitbox.center)
        screen.blit(self.car, self.car_rect)

    def get_reward(self):
        reward = 0
        if self.grid not in self.visited:
            self.visited.add(self.grid)
            reward += 5
        reward += abs(self.velocity)
        reward -= abs(self.front_back_diff)*5
        reward -= abs(self.left_right_diff)*2
        reward += self.death_pentalty
        self.death_pentalty = 0
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