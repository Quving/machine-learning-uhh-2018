import numpy as np
import plotter

###### TASK 1

### SETTINGS
database_filename = 'vaccination.csv'
plot_dir = 'plots'


### A 

# Read csv, use the header for names argumment to access the data more easily
data = np.genfromtxt(database_filename, delimiter=",", autostrip=True, names=True)

# Save the arrays into separate variables for convenience
gender = data['gender']
age = data['age']
height = data['height']
weight = data['weight']
residence = data['residence']
olderSiblings = data['olderSiblings']
knowsToRideABike = data['knowsToRideABike']
vacX = data['vacX']
diseaseX = data['diseaseX']
diseaseY = data['diseaseY']
diseaseZ = data['diseaseZ']

diseases = {
    'diseaseX': diseaseX.sum(),
    'diseaseY': diseaseY.sum(),
    'diseaseZ': diseaseZ.sum()
}

def distribution_of(data):
    return np.unique(data, return_counts=True)


def task1a():

    plotter.set_plot_dir(plot_dir)

    print('''
    ------------------------------------
    Task 1.A
    ------------------------------------
    GOAL: extract data from CSV, 
          determine number of boys / girls, age diseases and olderSiblings,
          Plot with Bar plots
    
    Results:''')

    # Overall data characteristics
    print("Number of samples: {}".format(len(gender)))
    
    # boys / girls, 
    classes, counts = distribution_of(gender)
    print("Gender: {}".format(np.vstack([classes,counts])))
    plotter.bars(counts, x_ticks=['female', 'male'], title="1A: Gender", normalize=False)

    # age 
    classes, counts = distribution_of(age)
    print("Ages: {}".format(np.vstack([classes,counts])))
    plotter.bars(counts, classes, title="1A: Age distribution", normalize=False)

    # knows to ride a bike
    print("Knows how to ride a bike: {}".format(knowsToRideABike.sum()))
    plotter.bool_bars(knowsToRideABike, title="1A: Knows how to ride a bike", normalize=False)

    # vaccines for disease x
    print("vacX: {}".format(vacX.sum()))
    plotter.bool_bars(vacX, title="1A: Vaccined against disease X", normalize=False)

    # diseases
    print("diseaseX: {}".format(diseaseX.sum()))
    print("diseaseY: {}".format(diseaseY.sum()))
    print("diseaseZ: {}".format(diseaseZ.sum()))
    plotter.bars(np.array(list(diseases.values())), np.array(list(diseases.keys())), title="1A: Disease distribution", normalize=False)

    # olderSiblings
    classes, counts = distribution_of(olderSiblings)
    print("OlderSiblings: {}".format([classes,counts]))
    plotter.bool_bars(olderSiblings, title="1A: Number of older siblings", normalize=False)


### B: Marginal properties
# GOAL: Calculate empirical probability for things.
# ...having a vaccination against disease X
# ...living on the country side
# ...having at least one older sibling

def marginal_probability(rows,rows_with_a):
    return len(rows_with_a) / len(rows)

def task1b():
    print('''
    ------------------------------------
    Task 1.B
    ------------------------------------
    GOAL: Calculate empirical probability for...
          ...having a vaccination against disease X
          ...living on the country side
          ...having at least one older sibling
          
          Calculate a variable diseaseYZ and 
          calculate the empirical probability for it.

    Results:''')


    # ...having a vaccination against disease X
    diseaseX_prob = marginal_probability(diseaseX, diseaseX[diseaseX[:] == 1])
    print("Empirical probability of disease X: {}%".format(diseaseX_prob*100))


    # ...living on the country side
    residence_prob = marginal_probability(residence, residence[residence[:] == 1])
    print("Empirical probability of living on the country side: {}%".format(residence_prob*100))

    # ...having at least one older sibling
    olderSiblings_prob = marginal_probability(olderSiblings, olderSiblings[olderSiblings[:] > 1])
    print("Empirical probability of having > 1 older siblings: {}%".format(olderSiblings_prob*100))


### C: Preprocessing
# GOAL: Calculate probabilities with preprocessing functions for
#       ... being X > 1m
#       ... being X > 40kg


# Write specific preprocessing functions for 
# ... being X > 1m
# ... being X > 40kg
# Calculate them
# Write combination function

### D: Conditional Probabilities
# 


### E

if __name__ == "__main__":
    # task1a()
    task1b()