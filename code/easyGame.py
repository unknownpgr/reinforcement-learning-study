import numpy as np
import random

WID = 84
HEI = 84


def rc(x, y, w, h):
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x > w:
        x = w
    if y > h:
        y = h
    return int(x), int(y)


class EasyGame:
    def __init__(self):
        self.initGame()

    def initGame(self):
        self.player = {'x': int(WID/2), 'y': 10, 'w': 8}
        self.time = 0
        # for time, let 1 tick be 10 ms.
        # Therefore, 100tick = 1s.

        self.stones = []
        self.stoneSize = 6
        self.gameEnd = False

    def step(self, userInput=[]):

        if self.gameEnd:
            self.initGame()

        # Shorten variable
        player = self.player

        # Game screen
        img = np.zeros([HEI, WID, 1], np.float32)

        # Move player by userInput
        action = np.argmax(userInput)
        if action == 0:
            player['x'] -= 2
        elif action == 1:
            pass
        else:
            player['x'] += 2

        # Draw player on game screen
        l, t = rc(player['x']-player['w']/2, 0, WID, HEI)
        r, b = rc(player['x']+player['w']/2, 0, WID, HEI)
        img[player['y'], l:r, :] = 1

        # Generate stone
        if self.time % 50 == 0:
            if self.time % 100 == 0:
                self.stones.append({'x': random.randint(0, WID), 'y': HEI})
            else:
                self.stones.append({'x': player['x'], 'y': HEI})

        # Move stone and check collision
        for stone in self.stones:
            stone['y'] -= .5
            if stone['y'] < 0:
                self.stones.remove(stone)
            elif abs(stone['x']-player['x']) < (self.stoneSize + player['w'])/2 and\
                    abs(stone['y']-player['y']) < (self.stoneSize)/2:
                gameEnd = True
                break

            # Draw stones
            l, t = rc(stone['x']-self.stoneSize/2,
                      stone['y'] + self.stoneSize/2, WID, HEI)
            r, b = rc(stone['x']+self.stoneSize/2,
                      stone['y'] - self.stoneSize/2, WID, HEI)
            img[b:t, l:r, :] = 1
        else:
            gameEnd = False

        # Limit position of player
        if player['x'] < 0 or player['x'] > WID:
            gameEnd = True

        self.time += 1

        return img, gameEnd, self.time


try:
    import time
    import cv2

    myGame = EasyGame()
    while True:
        action = [0, 1, 0]
        key = cv2.waitKey(10)
        if key == 100:
            action = [0, 0, 1]
        if key == 97:
            action = [1, 0, 0]
        img, gameEnd, time = myGame.step(action)
        if gameEnd:
            break
        cv2.imshow('EasyGame', img[::-1, :, :]*255)
    print("Your score =", myGame.time)

except Exception:
    print('Easygame module mode')
