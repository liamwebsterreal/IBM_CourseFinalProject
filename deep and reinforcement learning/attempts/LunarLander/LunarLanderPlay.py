import gym
import time

import pygame
import keyboard

env = gym.make("LunarLander-v2")
env.reset()
env.render()

time.sleep(5)
print(3)
time.sleep(1)
print(2)
time.sleep(1)
print(1)

while True:

    if keyboard.is_pressed('a'):
        env.step(3)
        env.render()
    elif keyboard.is_pressed('s'):
        env.step(2)
        env.render()
    elif keyboard.is_pressed('d'):
        env.step(1)
        env.render()
    elif keyboard.is_pressed('q'):
        break
    else:
        env.step(0)
        env.render()

pygame.quit()