The individuals are located on a map of size 12*12.
The populations has been created depending on the position of individuals. The map is squared and a population is assigned to each individual according to the square in which it is located.




genome.csv : a matrix of size n x L with n number of individuals and L number of locus
var_current.csv : a matrix of size n x 2. The first column corresponds to value of VAR1 and the second to value of VAR2
var_futur.csv : a matrix of size n x 2. The first column corresponds to value of VAR1 and the second to value of VAR2
pop.csv : a vector of size n. Each element corresponds to the population of the i-th individual.
position.csv : a matrix of size n x 2. The x and y position of individuals
mutationm2.csv : The position of mutation related to local adaptation to VAR1
mutationm3.csv : The position of mutation related to local adaptation to VAR2
fitness_current.csv : A vector of size n. Each element corresponds to the population of the i-th individual in the current environment.
fitness_futur.csv : A vector of size n. Each element corresponds to the population of the i-th individual in the future environment.
