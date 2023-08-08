if __name__ == "__main__":
    run(n_steps=number_steps, n_configs=number_configs)
    print("Sum of Q_value_matrix before= ", np.sum(Q_value_matrix))
    Q_value_matrix = np.loadtxt("Q_value_matrix.csv", delimiter=",")
    print("Sum of Q_value_matrix after= ", np.sum(Q_value_matrix))
