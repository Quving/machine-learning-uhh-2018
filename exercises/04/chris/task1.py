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
    plotter.bars(counts, x_ticks=['female', 'male'], title="1A: Gender", normalize=False, show=False, filename="1a_gender")

    # age 
    classes, counts = distribution_of(age)
    print("Ages: {}".format(np.vstack([classes,counts])))
    plotter.bars(counts, ['0 - 2', '3 - 6', '7 - 10', '11 - 13', '14 - 17'], title="1A: Age distribution", normalize=False, show=False, filename="1a_age")

    # residence 
    classes, counts = distribution_of(residence)
    print("residences: {}".format(np.vstack([classes,counts])))
    plotter.bars(counts, ['country side', 'town', 'small city', 'big city'], title="1A: Residence distribution", normalize=False, show=False, filename="1a_residence")

    # knows to ride a bike
    print("Knows how to ride a bike: {}".format(knowsToRideABike.sum()))
    plotter.bool_bars(knowsToRideABike, title="1A: Knows how to ride a bike", normalize=False, show=False, filename="1a_knowstorideabike")

    # vaccines for disease x
    print("vacX: {}".format(vacX.sum()))
    plotter.bool_bars(vacX, title="1A: Vaccined against disease X", normalize=False, show=False, filename="1a_vacX")

    # diseases
    print("diseaseX: {}".format(diseaseX.sum()))
    print("diseaseY: {}".format(diseaseY.sum()))
    print("diseaseZ: {}".format(diseaseZ.sum()))
    plotter.bars(np.array(list(diseases.values())), np.array(list(diseases.keys())), title="1A: Has / had the disease X/Y/Z", normalize=False, show=False, filename="diseases")

    # olderSiblings
    classes, counts = distribution_of(olderSiblings)
    print("OlderSiblings: {}".format([classes,counts]))
    plotter.bool_bars(olderSiblings, title="1A: Number of older siblings", normalize=False, show=False, filename="1a_olderSiblings")


### B: Marginal properties
# GOAL: Calculate empirical probability for things.
# ...having a vaccination against disease X
# ...living on the country side
# ...having at least one older sibling

# Calculates the marginal probability by two given data sets
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
def task1c():
    print('''

    ------------------------------------
    Task 1.C
    ------------------------------------
    GOAL: Calculate probabilities with preprocessing functions for
        ... being X > 1m
        ... being X > 40kg

    Results:''')

    isTallerThan1Meter = (height[:] > 100).astype(np.float32)
    isHeavierThan40Kilograms = (weight[:] > 40).astype(np.float32)

    # Combine diseaseY and diseaseZ and calculate empirical probability
    diseaseYZ = diseaseY + diseaseZ
    diseaseYZ_prob = marginal_probability(diseaseYZ, diseaseYZ[diseaseYZ[:] == 1])
    print("Empirical probability of disease Y or Z: {}%".format(diseaseYZ_prob*100))


### D: Conditional Probabilities
def conditional_probability(rows_with_a, rows_with_b, label_a="", label_b=""):
    return (len(rows_with_a) + len(rows_with_b)) / len(rows_with_b)

def conditional_probabilities(rows_with_a, rows_with_b, conditions_for_b, label_a="", label_b=""):

    print('Calculating probability for combination {} and {} = {}.'.format(label_a,label_b, conditions_for_b))

    probabilities = np.array([])

    for condition in conditions_for_b:
        probab = conditional_probability(rows_with_a, rows_with_b[rows_with_b[:] == condition], label_a=label_a, label_b=label_b)
        print('Probability for condition {} == {}: {}'.format(label_b, condition, probab))
        probabilities = np.append(probabilities, probab)

    return probabilities

def task1d():

    print('''

    ------------------------------------
    Task 1.D
    ------------------------------------
    GOAL: Calculate conditional probabilities for 
          – Pˆ(diseaseX | vacX = 0/1)
          – Pˆ(vacX | diseaseX = 0/1)
          – Pˆ(diseaseY | age = 1/2/3/4)
          – Pˆ(vacX | age = 1/2/3/4)
          – Pˆ(knowsT oRideABike | vacX = 0/1)

    Results:''')

    # – Pˆ(diseaseX | vacX = 0/1)
    conditional_probabilities(diseaseX, vacX, [0,1], label_a='diseaseX', label_b='vacX')

    # – Pˆ(vacX | diseaseX = 0/1)
    conditional_probabilities(vacX, diseaseX, [0,1], label_a='vacX', label_b='diseaseX')

    # – Pˆ(diseaseY | age = 1/2/3/4)
    probab_diseaseY_age = conditional_probabilities(diseaseY, age, [1,2,3,4], label_a='diseaseY', label_b='age')
    plotter.lines(probab_diseaseY_age, title="Probability: Had diseaseY on ages [1,2,3,4]", filename="1d_diseaseY_age")

    # – Pˆ(vacX | age = 1/2/3/4)
    probab_vacX_age = conditional_probabilities(vacX, age, [1,2,3,4], label_a='vacX', label_b='age')
    plotter.lines(probab_vacX_age, title="Probability: Is vaccined against diseaseX on ages [1,2,3,4]", filename="1d_vacX_age")

    # – Pˆ(knowsToRideABike | vacX = 0/1)
    conditional_probabilities(knowsToRideABike, vacX, [0,1], label_a='knowsToRideABike', label_b='vacX')

### E

if __name__ == "__main__":
    task1a()
    task1b()
    task1c()
    task1d()