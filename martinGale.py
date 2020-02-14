#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 21:06:21 2019

@author: itamar
"""

import random
import matplotlib
import matplotlib.pyplot as plt
#
import time

def rollDice():
    roll = random.randint(1,100)

    if roll == 100:
        return False
    elif roll <= 30:
        return False
    elif 100 > roll >= 30:
        return True


def doubler_bettor(funds,initial_wager,wager_count):
    value = funds
    wager = initial_wager
    wX = []
    vY = []
    currentWager = 1

    # since we'll be betting based on previous bet outcome #
    previousWager = 'win'

    # since we'll be doubling #
    previousWagerAmount = initial_wager

    while currentWager <= wager_count:
        if previousWager == 'win':
          # print('we won the last wager, yay!')
            if rollDice():
                value += wager
               #print(value)
                wX.append(currentWager)
                vY.append(value)
            else:
                value -= wager  
                previousWager = 'loss'
#               print(value)
                previousWagerAmount = wager
                wX.append(currentWager)
                vY.append(value)
                if value < 0:
                    wX.append(currentWager)
                    vY.append(value)
                    print('broke')
             #      print('went broke after',currentWager,'bets')
                    #currentWager += 10000000000000000
                    return 1 , wager
                    
        elif previousWager == 'loss':
           #print('we lost the last one, so we will be super smart & double up!')
            if rollDice():
                wager = previousWagerAmount * 2
              # print('we won',wager)
                value += wager
             #  print(value)
                wager = initial_wager
                previousWager = 'win'
                wX.append(currentWager)
                vY.append(value)
            else:
                wager = previousWagerAmount * 2
             #  print('we lost',wager)
                value -= wager
                if value < 0:
                    wX.append(currentWager)
                    vY.append(value)
                    print('broke')
                    return 1 , wager
            #   print(value)
                previousWager = 'loss'
                previousWagerAmount = wager
                wX.append(currentWager)
                vY.append(value)
                if value < 0:
                    wX.append(currentWager)
                    vY.append(value)
                    print('broke')
                    return 1 , wager
                
                #   print('went broke after',currentWager,'bets')
                    

        currentWager += 1

   #print(value)
    plt.plot(wX,vY)
    return 0 , wager

count = 0
wags = []
wag = 0
for i in range (100):
    mid , wag = doubler_bettor(1000,1,1000)
    count = mid + count
    wags.append(wag)
plt.show()
print('wags:', wags)
print('precision:', (100-count)/100)




'''
Simple bettor, betting the same amount each time.

def simple_bettor(funds,initial_wager,wager_count):
    value = funds
    wager = initial_wager
    wX = []
    vY = []
    currentWager = 1
    while currentWager <= wager_count:
        if rollDice():
            value += wager
            wX.append(currentWager)
            vY.append(value)
        else:
            value -= wager
            wX.append(currentWager)
            vY.append(value)
        currentWager += 1
    plt.plot(wX,vY)
x = 0


while x < 100:
    simple_bettor(10000,100,100)
    x += 1

plt.ylabel('Account Value')
printlabel('Wager Count')
plt.show()
'''
		