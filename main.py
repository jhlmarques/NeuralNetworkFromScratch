import neuralnetwork as nn

# Trains a neural network to learn XOR
if __name__ == "__main__":

    # /// Training configs ///
    training_epochs = 5000
    learning_rate = 0.2

    # Batch size = 4
    training_batches = [([[1, 0], [0, 1], [0, 0], [1, 1]], [[1], [1], [0], [0]])]

    # Batch size = 3
    # training_batches = [([[1, 0], [0, 1], [0, 0]], [[1], [1], [0]]),
    #                     ([[1, 1]], [[0]])]

    # Batch size = 2
    # training_batches = [([[1, 0], [0, 1]], [[1], [1]]),
    #                     ([[1, 1], [0, 1]], [[0], [0]])]

    # Batch size = 1
    # training_batches =  [([[1, 0]], [[1]]),
    #                     ([[0, 1]], [[1]]),
    #                     ([[0, 0]], [[0]]),
    #                     ([[1, 1]], [[0]])]

    # /// Logging configs ///

    # Should the network's functions be logged when an epoch is logged?
    log_epoch_should_log_functions = False
    # Should the batch and network outputs be compared when logging?
    log_epoch_should_compare_outputs = True
    # How many epochs it takes for the network to print out a log?
    log_epoch_interval = 500
    # Auxiliary variables
    log_current_epoch = True
    log_training_functions = log_epoch_should_log_functions
    log_comparison = log_epoch_should_compare_outputs

    # Network instantiation
    Network = nn.NeuralNetwork(2, len(training_batches), learn_bias=True)
    Network.append_layer(2, 0, nn.sigmoid, nn.d_sigmoid)
    Network.append_layer(1, 0, nn.sigmoid, nn.d_sigmoid)

    print(f"-----------< EPOCH {0:04} >-----------")
    for i in range(training_epochs + 1):
        for batch in training_batches:
            Network.change_batch(batch[0], batch[1])
            Network.train_batch(learning_rate, log_training_functions, log_comparison)

        if (i+1) % log_epoch_interval == 0:
            log_current_epoch = True
            log_training_functions = log_epoch_should_log_functions
            log_comparison = log_epoch_should_compare_outputs
            print(f"-----------< EPOCH {(i+1):04} >-----------")
        elif log_current_epoch:
            log_current_epoch = False
            log_training_functions = False
            log_comparison = False
            print(f"Loss:\t{sum(Network.losses)}\n")
