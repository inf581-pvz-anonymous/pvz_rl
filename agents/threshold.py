# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:25:15 2021

@author: Lukas Frank
"""
import math
import numpy as np
import warnings

class Threshold():
    """
    Generate sequences of epsilon thresholds.
    
    :param seq_length: int, length of epsilon sequence = number of epsilons to draw
    
    :param start_epsilon: float, value to start with
    
    :param end_epsilon (optional): float, value to end with. If None, return constant
        sequence of value start_epsilon. Default: None.
    
    :param interpolation (optional): string, interpolation method:\n
        either 'linear', 'exponential' or 'sinusoidal'. Default: 'linear'.
        Reference: http://cs231n.stanford.edu/reports/2017/pdfs/616.pdf
    
    :param periods: int, number of periods for sinusoidal sequence. Default: 10.
        ...
    """
    def __init__(self, seq_length, start_epsilon, end_epsilon=None, 
                 interpolation='linear', periods=10):
        self.seq_length = seq_length
        self.start_epsilon = start_epsilon
        assert interpolation in ['linear', 'exponential', 'sinusoidal'], "interpolation argument invalid. Must be 'linear', 'exponential', 'sinusoidal' or unspecified."
        self.interpolation = interpolation
        if end_epsilon is None:
            self.end_epsilon = start_epsilon
            # set to linear to deliver constant sequence of epsilons.
            self.interpolation = 'linear'
        else:
            self.end_epsilon = end_epsilon
        self.periods = periods
        
    def epsilon(self, index=None):
        """Return sequence or element of sequence of epsilons as specified
        
        :param index (optional): index of sequence element to be returned. If None, return
            full sequence. Default: None.\n
        
        :return: array-like with shape (self.seq_length) or a single float value.
        """
        epsilon= None

        if self.interpolation == 'linear':
            epsilon = self._linear(index)
            
        elif self.interpolation == 'exponential':
            epsilon = self._exponential(index)
            
        elif self.interpolation == 'sinusoidal':
            epsilon = self._sinusoidal(index,self.periods)

        return epsilon
    
    def _linear(self, index):
        """Calls linear calculation method depending on whether index is given or not."""
        if index is not None: # return only one epsilon
            self._check_index_length(index)
            return self._linear_point(index)
        else:
            return self._linear_sequence()
     
    def _exponential(self, index):
        """Calls exponential calculation depending on whether index is given or not."""
        if index is not None: # return only one epsilon
            self._check_index_length(index)
            return self._exponential_point(index)
        else:
            return self._exponential_sequence()

    def _sinusoidal(self, index, periods):
        """Calls sinusoidal calculation depending on whether index is given or not."""
        if index is not None: # return only one epsilon
            self._check_index_length(index)
            return self._sinusoidal_point(index, mini_epochs=periods)
        else:
            return self._sinusoidal_sequence(mini_epochs=periods)
        
    def _linear_sequence(self):
        """Computes linear sequence"""
        return np.linspace(start=self.start_epsilon, 
                           stop=self.end_epsilon, 
                           num=self.seq_length).tolist()
    
    def _linear_point(self, index):
        """Computes a single point by linear interpolation"""
        return self.start_epsilon + (self.end_epsilon-self.start_epsilon)/(self.seq_length-1) * index
    
        
    def _exponential_sequence(self):
        """Computes exponential sequence"""
        decay_rate = (self.end_epsilon/self.start_epsilon)**(1/(self.seq_length-1))
        return [(self.start_epsilon * decay_rate**i) for i in range(self.seq_length)]
                

    def _exponential_point(self, index):
        """Computes a single point by exponential interpolation"""
        decay_rate = (self.end_epsilon/self.start_epsilon)**(1/(self.seq_length-1))
        return self.start_epsilon * decay_rate**index
        
    def _sinusoidal_sequence(self, mini_epochs):
        """Computes sinusoidal sequence.
        
        Reference: http://cs231n.stanford.edu/reports/2017/pdfs/616.pdf \n
        
        :param mini_epochs (optional): int, number of oscillations in sequence.
        """
        decay_rate = (self.end_epsilon/self.start_epsilon)**(1/(self.seq_length-1))
        return [(self.start_epsilon * decay_rate**i * 0.5*(1+np.cos(2*math.pi*i*mini_epochs/(self.seq_length-1)))) 
                for i in range(self.seq_length)]

    def _sinusoidal_point(self, index, mini_epochs):
        """Computes a single point by sinusoidal interpolation.
        
        Reference: http://cs231n.stanford.edu/reports/2017/pdfs/616.pdf \n
        
        :param mini_epochs (optional): int, number of oscillations in sequence.
        """
        decay_rate = (self.end_epsilon/self.start_epsilon)**(1/(self.seq_length-1))
        return self.start_epsilon * decay_rate**index * 0.5*(1+np.cos(2*math.pi*index*mini_epochs/(self.seq_length-1)))
    
    def _check_index_length(self, index):
        """Check whether index is in sequence."""
        if index >= self.seq_length:
            warnings.warn("threshold.epsilon(index): index > seq_length. There might be some unintended results.")
            
        