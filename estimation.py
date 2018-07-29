import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize

# model
def probabilityCorrect(x, x_coordinate):
    return 0.5 + (0.5 - x[0])*(1 - np.exp(-(x_coordinate/x[1])**x[2]))

# cost function
def cost_ML(x, x_coordinate, y_coordinate, number_of_data):       
    # compute cost
    c = 0
    for n in np.arange(len(number_of_data)):
        # model output 
        p = probabilityCorrect(x, x_coordinate[n])

        # cost as the negative likelihood
        if 0 < p <= 1:
            c += -number_of_data[n]*(y_coordinate[n]*np.log(p) + (1 - y_coordinate[n])*np.log(1 - p))

    return c 

# data
x_coordinate = [0, 0.2, 0.4, 0.6, 0.8]
y_coordinate = [0.5, 0.81, 0.87, 0.92, 0.95]
number_of_data = [50, 45, 40, 35, 30]

# initial guess and boundary
x0 = [0, 0.2, 1]
bound = [(0, None),(0.0001, None),(0,None)]

# maximum likelihood
params_ML = optimize.minimize(cost_ML, x0, args=(x_coordinate, y_coordinate, number_of_data, ), method='l-bfgs-b', \
                        jac=None, bounds=bound, tol=None, callback=None,\
                        options={'disp': None, 'maxls': 20, 'iprint': -1,\
                                 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000,\
                                 'ftol': 2.220446049250313e-09, 'maxcor': 10,\
                                 'maxfun': 15000})

# compute variance explained
fitted_acc_ml = np.zeros(len(x_coordinate))
for s in np.arange(len(x_coordinate)):
    fitted_acc_ml[s] = probabilityCorrect(params_ML.x,x_coordinate[s])

varexp_ml = 1 - (np.var(fitted_acc_ml - y_coordinate)/np.var(y_coordinate))

print('variance explained: ' + str(varexp_ml))

# visualize
x_coordinates = np.linspace(0,0.8,100)
acc_ml = np.zeros(100)

for i in np.arange(100):
    acc_ml[i] = probabilityCorrect(params_ML.x,x_coordinates[i])

fig = plt.figure()

ax = fig.add_subplot(111)

ax.set_xlabel('x')
ax.set_ylabel('y')    
plt.plot(x_coordinate, y_coordinate, 'ko')
plt.plot(x_coordinates, acc_ml, '-b')
plt.show()
