import sys
import logging
import time
from iqoptionapi.stable_api import IQ_Option
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def login(verbose = False):
    
    if verbose:
        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')

    iq=IQ_Option("username","password")
    iq.change_balance("PRACTICE")
    return iq

def get_candles(iq,Actives):
    return iq.get_candles(Actives,60,1000,time.time())

