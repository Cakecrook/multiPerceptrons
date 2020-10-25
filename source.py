import sys
import numpy as np
import pandas as pd

def sigmoid(inp):
    return 1 / (1 + np.exp(-inp))

def sigmoid_der(inp):
    return sigmoid(inp) * (1 - sigmoid(inp))

def main():
    inputs = pd.read_csv(sys.argv[1])
    inputs = inputs.values.tolist()
    test_inputs = pd.read_csv(sys.argv[2])
    test_inputs = test_inputs.values.tolist()

    num_inputs = len(inputs) # 12664
    test_num_inputs = len(test_inputs)

    input_dimension = len(inputs[0]) # 785
    hidden_dimension = 5
    iterations = 5
    lr = 0.2

    target_values = []
    input_features = []
    test_target_values = []
    test_input_features = []

    for i in range(num_inputs):
        target_values.append(inputs[i][0])
        input_features.append(inputs[i][1:])
    for i in range(test_num_inputs):
        test_target_values.append(test_inputs[i][0])
        test_input_features.append(test_inputs[i][1:])

    for i in range(len(input_features)):
        for j in range(len(input_features[0])):
            input_features[i][j] /= 255
    for i in range(len(test_inputs)):
        for j in range(len(test_inputs[0])):
            test_inputs[i][j] /= 255
    
    ItoHweights = [[0] * (input_dimension - 1)] * hidden_dimension
    HtoOweights = [0] * hidden_dimension                     # 5
    hidden_biases = [0] * hidden_dimension                   # 5
    output_bias = np.random.uniform(-1, 1)

    for i in range(len(ItoHweights)):
        for j in range(len(ItoHweights[i])):
            ItoHweights[i][j] = np.random.uniform(-1, 1)
    for i in range(len(HtoOweights)):
        HtoOweights[i] = np.random.uniform(-1, 1)
    for i in range(len(hidden_biases)):
        hidden_biases[i] = np.random.uniform(-1, 1)

    for epoch in range(iterations): # for epoch in iterations(10000)
        for elem in range(num_inputs): # for image in 12664 inputs
            # Feedforward
            in_h = [0] * hidden_dimension
            out_h = [0] * hidden_dimension
            for node in range(hidden_dimension): # for node in 5
                for i in range(len(input_features[elem])): # for i in 784
                    in_h[node] += input_features[elem][i] * ItoHweights[node][i]
                in_h[node] += hidden_biases[node]
                out_h[node] = sigmoid(in_h[node])

            in_o = 0
            for i in range(hidden_dimension):
                in_o += out_h[i] * HtoOweights[i]
            in_o += output_bias
            out_o = sigmoid(in_o)

            # Backpropagation
            error = out_o - target_values[elem]

            dError_dOuto = error
            dOuto_dIno = sigmoid_der(out_o)
            delta_o = dError_dOuto * dOuto_dIno

            delta_h = [0] * hidden_dimension 

            for hidden_node in range(hidden_dimension):
                delta_h[hidden_node] = sigmoid_der(in_h[hidden_node])
                delta_h[hidden_node] = delta_h[hidden_node] * HtoOweights[hidden_node] * delta_o
        
            for hidden_node in range(hidden_dimension):
                HtoOweights[hidden_node] -= lr * delta_o * out_h[hidden_node]

            for hidden_node in range(hidden_dimension):
                for input_node in range(input_dimension - 1):
                    ItoHweights[hidden_node][input_node] -= lr * delta_h[hidden_node] * input_features[elem][input_node]

            for hidden_bias in range(len(hidden_biases)):
                hidden_biases[hidden_bias] -= lr * delta_h[hidden_bias]
            
            output_bias -= lr * delta_o

    print("[UPDATE] Training done, starting testing phase")
    accuracy = 0
    for elem in range(test_num_inputs):
        for node in range(hidden_dimension):
            in_h[node] = 0
            for i in range(len(test_inputs[elem][1:])):
                in_h[node] += test_inputs[elem][i + 1] * ItoHweights[node][i]
            in_h[node] += hidden_biases[node]
            out_h[node] = sigmoid(in_h[node])

        in_o = 0
        for i in range(hidden_dimension):
            in_o += out_h[i] * HtoOweights[i]
        in_o += output_bias
        out_o = sigmoid(in_o)

        if out_o >= 0.5:
            prediction = 1
        else:
            prediction = 0

        if prediction == test_target_values[elem]:
            accuracy += 1
    print("[COMPLETE] Accuracy = %", (accuracy/test_num_inputs) * 100, sep='')
    return

if __name__ == '__main__':
    main()