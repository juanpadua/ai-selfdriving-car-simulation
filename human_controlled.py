#MODULES
import pygame
import math

screen=pygame.display.set_mode((1920,1080))

car_w=150
car_h=125

class Car:
    def __init__(self,image,x,y):
        self.car=pygame.image.load(f'data/cars/{image}').convert_alpha()
        self.car=pygame.transform.scale(self.car,(car_w,car_h))
        self.rotated_car=self.car
        self.car_hitbox=pygame.Rect(x,y,car_w,car_h)
        self.velocity=0
        self.acceleration=0.3
        self.friction=0.95
        self.angle=0
        self.rotation=0
    
    def controls(self,keys):
        if keys[pygame.K_w]:
            self.velocity+=self.acceleration
        if keys[pygame.K_s]:
            self.velocity-=self.acceleration
        if keys[pygame.K_d]:
            self.angle-=self.rotation
        if keys[pygame.K_a]:
            self.angle+=self.rotation

        if self.velocity>2:
            self.rotation=1.5
        elif self.velocity<-2:
            self.rotation=-1.5
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

pygame.init()

car1=Car('car1.png',960,540)

running=True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running=False
    pygame.display.flip()
    screen.fill((64,64,64))
    keys=pygame.key.get_pressed()

    car1.controls(keys)
    car1.math()
    car1.draw()

pygame.quit()