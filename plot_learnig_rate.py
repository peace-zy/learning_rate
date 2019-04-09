import matplotlib.pyplot as plt
import numpy as np

def inv(itr, base_lr, gamma, power):
    y = base_lr * (1 + gamma * itr)**(-power)
    return y

def poly(itr, base_lr, max_iter, power):
    y = base_lr * (1 - itr / float(max_iter))**(power)
    return y

def sigmoid(itr, base_lr, gamma, stepsize):
    y = base_lr * (1 / (1 + np.exp(-gamma * (itr - stepsize))))
    return y

def exp(itr, base_lr, gamma):
    y = base_lr * gamma**itr
    return y

if __name__ == '__main__':
    base_lr = 0.01
    max_iter = 1200000

    '''
    #inv stagetory
    '''
    gamma = 0.0001
    power = 0.75
    itr = np.arange(0, max_iter, 100)
    y = inv(itr, base_lr, gamma, power)
    plt.figure(1)
    plt.title("inv stategory")
    plt.plot(itr, y)
    ##########################

    '''
    #poly stagetory
    '''
    power = 5 
    itr = np.arange(0, max_iter, 100)
    y = poly(itr, base_lr, max_iter, power)
    for i in range(0,len(y), 500):
        print(str(i) + " " + str(y[i]))
    plt.figure(2)
    plt.title("poly stategory")
    plt.plot(itr, y)
    ##########################

    '''
    #sigmoid stagetory
    '''
    gamma = -0.0001
    stepsize = 250000 
    itr = np.arange(0, max_iter, 100)
    y = sigmoid(itr, base_lr, gamma, stepsize)
    plt.figure(3)
    plt.title("sigmoid stategory")
    plt.plot(itr, y)
    ##########################

    '''
    #exp stagetory
    '''
    base_lr = 0.01
    gamma = 0.99 
    itr = np.arange(0, max_iter, 100)
    y = exp(itr, base_lr, gamma)
    plt.figure(4)
    plt.title("exp stategory")
    plt.plot(itr, y)
    plt.show()
    ##########################
