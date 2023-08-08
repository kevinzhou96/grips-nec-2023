number_steps= 200
number_configs= 2
learning_rate = 1/5
discount_factor = 999/1000
epsilon0 =0/10  #for making actions random
save_counter = 1  #save csv files if (number of initializing all agent) in {m * save_counter | m is natural number}
dawnload_counter = 20 #dawnload csv files if (number of initializing all agent) in {m * dawnload_counter | m is natural number}
Counter = 0  #initialize
use_data_of_needs_of_days = False
treat_level = True　#Whether to find the optimal unit price
　
competitors = [AgentForQLearning0726, SimpleOneShotAgent]


"""define Q_value_matrix"""
if True: #define from csv file
    Q_value_matrix = np.loadtxt("Q_value_matrix.csv", delimiter=",")
elif False: #all 0
    if treat_level:
        number_of_col = len(choice_of_quantity) * 2
    else:
        number_of_col = len(choice_of_quantity)
    Q_value_matrix = np.zeros((6006, number_of_col), dtype=int)
    np.savetxt("Q_value_matrix.csv", Q_value_matrix, delimiter=",")


"""avoid making proposals that exceed your needs"""
if False:
    decrease_value = 1000
    for row in range(6006):
        step = row % 21
        needs_value = ((row - step) // 21) % 13
        level = ((row - step - needs_value * 21) // 273) % 2
        n_of_partners = (row - step - needs_value * 21 - level * 273) // 546
        if needs_value > 2:
            if treat_level:
                col_size = len(choice_of_quantity) * 2
                for col in range(col_size):
                    selected_choice = col % len(choice_of_quantity)
                    if choice_of_quantity[selected_choice](needs_value - 2, n_of_partners + 1) > needs_value - 2:
                        #print("step, needs, level, n_of_partners = ", step, needs, level, n_of_partners)
                        Q_value_matrix[row, col] -= decrease_value
    np.savetxt("Q_value_matrix.csv", Q_value_matrix, delimiter=",")
