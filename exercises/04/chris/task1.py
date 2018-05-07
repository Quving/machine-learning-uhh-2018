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
    plotter.bars(np.array(list(diseases.values())), np.array(list(diseases.keys())), title="1A: Has / had the disease X/Y/Z", normalize=False, show=False, filename="1a_diseases")

    # olderSiblings
    classes, counts = distribution_of(olderSiblings)
    print("OlderSiblings: {}".format([classes,counts]))
    plotter.bars(counts, classes, title="1A: Number of older siblings", normalize=False, show=False, filename="1a_olderSiblings")


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

    global isTallerThan1Meter
    global isHeavierThan40Kilograms

    isTallerThan1Meter = (height[:] > 100).astype(np.float32)
    isHeavierThan40Kilograms = (weight[:] > 40).astype(np.float32)

    # Combine diseaseY and diseaseZ and calculate empirical probability
    global diseaseYZ
    global diseaseYZ_prob
    diseaseYZ = diseaseY + diseaseZ
    diseaseYZ[diseaseYZ[:] > 1] = 1

    diseaseYZ_prob = marginal_probability(diseaseYZ, diseaseYZ[diseaseYZ[:] == 1])
    print("Empirical probability of disease Y or Z: {}%".format(diseaseYZ_prob*100))


### D: Conditional Probabilities
def conditional_probability(rows_with_a_and_b, rows_with_b):
    return len(rows_with_a_and_b) / len(rows_with_b)

def conditional_probabilities(dataset_a, dataset_b, conditions_for_b, conditions_for_a=[1], label_a="", label_b=""):

    '''
    Calculates the contitional probabilities for two datasets A and B with CONDITIONS for A and CONDITIONS for B the following:
    1) For each CONDITION for A:
    1.1) For each CONDITION for B:
    1.1.1) Calculate the conditional probability for P(A = CONDITION | B = CONDITION)

    The return value is an array, containing all probabilities for all P(A = CONDITIONS | B = CONDITIONS)
    '''

    print('\nCalculating probability for combination: {}, given {} = {}.'.format(label_a,label_b, conditions_for_b))

    probabilities = np.array([])

    for condition_a in conditions_for_a:
        for condition_b in conditions_for_b:
            print("- Condition: A: {} == {}, B: {} == {} -------------------".format(label_a, condition_a, label_b, condition_b))

            # Prepare coditions by filtering datasets
            dataset_b_f = dataset_b[dataset_b[:] == condition_b]
            dataset_a_and_b_f = dataset_a[(dataset_a[:] ==condition_a) & (dataset_b[:] == condition_b)]

            probab = conditional_probability(dataset_a_and_b_f, dataset_b_f)
            print('Conditional probability: {}'.format(probab))
            probabilities = np.append(probabilities, probab)
    return probabilities


def conditional_probabilities_threefolded(dataset_a, dataset_b, dataset_c, conditions_for_b, conditions_for_c, conditions_for_a=[1], label_a="", label_b="", label_c=""):
    '''
    Calculates the conditional probabilities of P(A|B = CONDITIONS, C = CONDITIONS) by following:
    1) Filter dataset A and B by C = CONDITIONS
    2) For each CONDITIONS for C:
    2.1) For each CONDITIONS for B_f (filtered):
    2.1.1) Calculate the conditional probability for P(A_f|B_f = CONDITION)

    The result gets saved as a M x N matrix. Each row contains the probabilities for P(A | B = CONDITIONS)
    '''

    probabilities = np.array([])

    print("\nStarting threefolded conditional probabilities analysis.\n\nDataset: \n{} with {} items\n{} with {} items\n{} with {} items".format(label_a, len(dataset_a), label_b, len(dataset_b), label_c, len(dataset_c)))

    # For each group in C, we calculate the conditional probabilities for A and B with given conditions
    for condition_c in conditions_for_c:

        print("\nFilter group {} == {}".format(label_c, condition_c))

        # Get first the filter for the dataset c with condition_c to filter all other datasets beforehand
        filter_c = dataset_c[:] == condition_c

        # Filter datasets a and b to only have the data of the given data range
        dataset_a_f = dataset_a[filter_c]
        dataset_b_f = dataset_b[filter_c]

        # Calculate the conditional probabilities
        probabilities = np.append(probabilities, np.array([conditional_probabilities(dataset_a_f, dataset_b_f, conditions_for_b, conditions_for_a=conditions_for_a, label_a=label_a, label_b=label_b)]))


    probabilities = probabilities.reshape((int(len(probabilities)/len(conditions_for_b)),int(len(conditions_for_b))))
    print(probabilities)
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
    global probab_diseaseX_vacX
    probab_diseaseX_vacX = conditional_probabilities(diseaseX, vacX, [0,1], label_a='diseaseX', label_b='vacX')

    # – Pˆ(vacX | diseaseX = 0/1)
    conditional_probabilities(vacX, diseaseX, [0,1], label_a='vacX', label_b='diseaseX')

    # – Pˆ(diseaseY | age = 1/2/3/4)
    global probab_diseaseY_age
    probab_diseaseY_age = conditional_probabilities(diseaseY, age, [1,2,3,4], label_a='diseaseY', label_b='age')
    plotter.lines(probab_diseaseY_age, title="Probability: Had diseaseY on ages [1,2,3,4]", filename="1d_diseaseY_age", x_ticks=["0 - 2y", "3 - 6y", "7 - 10y", "11 - 13y"])

    # – Pˆ(vacX | age = 1/2/3/4)
    global probab_vacX_age
    probab_vacX_age = conditional_probabilities(vacX, age, [1,2,3,4], label_a='vacX', label_b='age')
    plotter.lines(probab_vacX_age, title="Probability: Is vaccined against diseaseX on ages [1,2,3,4]", filename="1d_vacX_age", x_ticks=["0 - 2y", "3 - 6y", "7 - 10y", "11 - 13y"])

    # – Pˆ(knowsToRideABike | vacX = 0/1)
    conditional_probabilities(knowsToRideABike, vacX, [0,1], label_a='knowsToRideABike', label_b='vacX')

### E
def task1e():

    print('''

    ------------------------------------
    Task 1.E
    ------------------------------------
    GOAL: Calculate Pˆ(diseaseYZ | vacX = 0/1) and compare it to Pˆ(diseaseX|vacX = 0/1). 
          What do you conclude from these results? 
    
          Now, condition additionally on age and calculate Pˆ(diseaseY Z | vacX = 0/1, age = 1/2/3/4).
          How sure are you that your estimates for P (diseaseY Z | vacX = 0/1, age = 1/2/3/4) are accurate? 
          What does this depend on?

          Plot Pˆ(diseaseY Z = 1 | vacX = 0, age = 1/2/3/4) and Pˆ(diseaseY Z = 1 | vacX = 1, age = 1/2/3/4) as two lines in one figure with age on the x-axis and the probability on the y-axis.
          What do you conclude from your plot?

    Results:
    ''')

    # Calculate Pˆ(diseaseY Z | vacX = 0/1) and compare it to Pˆ(diseaseX|vacX = 0/1). 
    # What do you conclude from these results?
    global probab_diseaseYZ_vacX
    probab_diseaseYZ_vacX = conditional_probabilities(diseaseYZ, vacX, [0,1], label_a='diseaseYZ', label_b='vacX')
    print("\n                          vacX == 0    vacX == 1\ndiseaseYZ & vacX probab: {}\ndiseaseX & vacX probab:  {}".format(probab_diseaseYZ_vacX, probab_diseaseX_vacX))
    plotter.lines(probab_diseaseYZ_vacX, savefig=False, stackableplot=True, label="diseaseYZ & vacX")
    plotter.lines(probab_diseaseX_vacX, title="Conditional probability of: got disease Y OR Z and is vaccined against disease X\nvs. got disease X and is vaccined against disease X", filename="1e_diseaseYZ_diseaseX_vacX_comparison", label="diseaseX & vacX", clear=False)

    print("Result: the probability for diseaseYZ and diseaseX in combination with vacX is nearly the same: both groups have a lower probability to get sick with diseases Y or Z or X given the fact that there are vaccined against diseaseX.")

    # Now, condition additionally on age and calculate Pˆ(diseaseYZ | vacX = 0/1, age = 1/2/3/4).
    probab_diseaseYZ_vacX_age = conditional_probabilities_threefolded(diseaseYZ, vacX, age, [0,1], [1,2,3,4], label_a="diseaseYZ", label_b="vacX", label_c="age")
    
    # How sure are you that your estimates for P (diseaseYZ | vacX = 0/1, age = 1/2/3/4) are accurate? 
    print('''
    Q: How sure are you that your estimates for P (diseaseYZ | vacX = 0/1, age = 1/2/3/4) are accurate? 
       What does this depend on?
    A: We assume that the data is interpretable because we assumed that the age groups 1/2/3/4 are distinguished from each other. So we don't have any dataset which is multiple times used for calculations.''')

    # Plot Pˆ(diseaseY Z = 1 | vacX = 0, age = 1/2/3/4) and Pˆ(diseaseY Z = 1 | vacX = 1, age = 1/2/3/4) as two lines in one figure with age on the x-axis and the probability on the y-axis. 
    probab_diseaseYZ1_vacX0_age = conditional_probabilities_threefolded(diseaseYZ, vacX, age, [0], [1,2,3,4], label_a="diseaseYZ", label_b="vacX", label_c="age")
    probab_diseaseYZ1_vacX1_age = conditional_probabilities_threefolded(diseaseYZ, vacX, age, [1], [1,2,3,4], label_a="diseaseYZ", label_b="vacX", label_c="age")

    probab_diseaseYZ1_vacX0_age = probab_diseaseYZ1_vacX0_age.flatten()
    probab_diseaseYZ1_vacX1_age = probab_diseaseYZ1_vacX1_age.flatten()

    print("Probability results, summarized for disease Y or Z, given vaccined == [0,1] against disease X in age groups [1,2,3,4]:\nvacX == 0: {}\nvacX == 1: {}".format(probab_diseaseYZ1_vacX0_age, probab_diseaseYZ1_vacX1_age))

    plotter.lines(probab_diseaseYZ1_vacX0_age, savefig=False, stackableplot=True, label="diseaseYZ = 1 & vacX = 0", x_ticks=[1,2,3,4])
    plotter.lines(probab_diseaseYZ1_vacX1_age, title="Conditional probability of:\ngot disease YZ, is / is not vaccined against vacX and in age of [1,2,3,4]", xlabel="age", ylabel="probability", savefig=True, filename="1e_diseaseYZ_diseaseX_vacX_age_comparison", label="diseaseYZ = 1 & vacX = 1", clear=False, x_ticks=["0 - 2y", "3 - 6y", "7 - 10y", "11 - 13y"], legend=True)

    # What do you conclude from your plot?
    print('''
    Q: What do you conclude from your plot?
    A: The probability of getting disease Y or Z with vaccination against disease X is dramatically low and constant over all age groups. Also, the risk of getting one of both diseases increases by the age.
    ''')


if __name__ == "__main__":
    
    task1a()
    task1b()
    task1c()
    task1d()
    task1e()