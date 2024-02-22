# Naive Bayesian Classification



# Sets up the data input into lists and runs program
def main():

    file1 = open("pdf.txt", 'r')
    bird_array = dict()
    plane_array = dict()
    i = 0
    j = 0
    lines = file1.readlines()
    line = lines[0]
    line = line.strip('\n')
    x = line.split(',')

    # Creates dictionary of probabilities where each index is a velocity
    # increasing by 0.5 from 0 -> 200 and at each index there is the
    # likelihood of the bird or plane traveling at that speed
    j = 0
    for i in range (0, 400):
        bird_array[j] = float(x[i])
        j += 0.5
    line = lines[1]
    line = line.strip('\n')
    x = line.split(',')
    i = 0
    for j in range (0, 400):
        plane_array[i] = float(x[j])
        i += 0.5

    # Creates a nested list of all of the observation data
    file2 = open("data.txt", 'r')
    data_array = []
    i = 0
    empty_list = []
    while i < 10:
        data_array.append([])
        i += 1
    lines = file2.readlines()
    y = 0
    for line in lines:
        line = line.strip('\n')
        x = line.split(',')
        for k in range (0, 300):
            if x[k] != 'NaN':
                new = float(x[k])
                new = round(new * 2) / 2
                data_array[y].append(new)
        y += 1

    # Runs program
    result = run (bird_array, plane_array, data_array)
    pretty_print(result)
    file2.close()
    file1.close()

# Helper function to print out the result
def pretty_print (result):
    for i in range(0,10):
        print("[", i + 1, "]: ", result[i])

# The main implementation of the naive Bayesian algorithm             
def run (bird_array, plane_array, data_array):

    classification = []
    b_plane = []
    b_bird = []

    # Creates two nested lists to store the probabilities for each
    # classifier for each observation
    for h in range (0, 10):
        b_plane.append([])
        b_bird.append([])

    # Starting at the first data point in each observation, the algorithm
    # multiplies the probability of each classifier at the first data
    # point and adds it to the nested lists
    for q in range (0, 10):
        plane = plane_array[data_array[q][0]] * 0.9
        b_plane[q].append(plane)
        bird = bird_array[data_array[q][0]] * 0.9
        b_bird[q].append(bird)

        # Normalizing the initial probabilities
        nums = plane + bird
        b_plane[q][0] = (b_plane[q][0] / nums)
        b_bird[q][0] = (b_bird[q][0] / nums)

        # For the rest of the data points in each observation, the algorithm
        # determines the classification probabilities by multiplying the initial
        # conditional probability by the sum of the transitional probability
        # multiplied by (for each classifier) the estimated probability up to
        # this point. 
        for x in range (1,len(data_array[q])):
            plane = b_plane[q][-1] + plane_array[data_array[q][x]] * \
                    (0.9 * b_plane[q][-1] + 0.1 * b_bird[q][-1])
            b_plane[q].append(plane)
            bird = b_bird[q][-1] + bird_array[data_array[q][x]] * \
                   (0.9 * b_bird[q][-1] + 0.1 * b_plane[q][-1])
            b_bird[q].append(bird)

            # Data is normalized 
            total_sum = plane + bird
            for i in range (0, len(b_plane[q])):
                if b_plane[q][i] != 0:
                    b_plane[q][i] = (b_plane[q][i] / total_sum)
            bird_sum = sum(b_bird[q])
            for j in range (0, len(b_bird[q])):
                if b_bird[q][i] != 0:
                    b_bird[q][j] = (b_bird[q][j] / total_sum)

    # This part determines the final classification, claiming that it cannot be
    # determined if the probabilities are within 10% of each other. 
    for x in range (0, 10):
        plane_num = b_plane[x][-1]
        bird_num = b_bird[x][-1]

        if plane_num > (bird_num + 0.10):
            classification.append("Aircraft = " + str(plane_num))
        elif bird_num > (plane_num + 0.05):
            classification.append("Bird = " + str(bird_num))
        else:
            classification.append("Could not be determined. ")
    
    return classification