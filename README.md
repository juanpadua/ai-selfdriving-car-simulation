# AI Self-Driving Car Simulation

## Overview

This project is an AI-powered self-driving car simulation developed using Python, Pygame, and NEAT (NeuroEvolution of Augmenting Topologies). The main aim of the project is to train a virtual car to drive on a track without human control. The car uses sensors to detect the track boundaries and a neural network decides how to control its speed and steering. Through NEAT’s evolutionary learning process, different neural networks are tested, evaluated using a fitness/reward system, and evaluated over multiple generations. This demonstrates how artificial intelligence, neural networks, and evolutionary algorithms can be applied to autonomous driving.

---

## Features

- We have an improvised good car sensing system which makes the AI car prefer to go straight in a road, and effectively turn when a car comes in the opposite direction.
- Users can create their own maps by drawing them in a different app (please note that the boundaries should be white).
- Whilst training our pgm stores the best model from each generation in cache folder to select the absolute best model of the lot.
- We have an interactive GUI for easy control and navigation

---

## Training
<p align="center">
  <img src="https://github.com/user-attachments/assets/39c30676-d035-41a8-aacb-a3ef779f82b6" width="300" alt="Training View"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e662d2ae-3944-408c-a086-770a5a521c76" width="480" alt="Training Stats"/>
  <img src="https://github.com/user-attachments/assets/9d9258a1-603c-4a1c-ae80-5b94c7cbbb38" width="450" alt="Fitness Graph"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/057b7cc8-f7a0-4585-b479-ad76459076e1" width="370" alt="Generation Progress"/>
</p>

---

## Simulation
<p align="center">
  <img src="https://github.com/user-attachments/assets/522878d1-53c5-4c16-946e-5e4da11970aa" width="200" alt="Sim 1"/>
  <img src="https://github.com/user-attachments/assets/c0e4075c-ac77-470f-b08f-6a3327a097b3" width="200" alt="Sim 2"/>
  <img src="https://github.com/user-attachments/assets/83f33826-800f-41ee-ab46-9b84b00638d5" width="200" alt="Sim 3"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/12df28a7-b7ca-4673-9a52-37ae666366b0" width="400" alt="Final Simulation"/>
</p>


---
