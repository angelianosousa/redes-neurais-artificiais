from use_csv import load_data_from_csv
from one_hot_crypt import one_hot_encode, one_hot_decode

import csv
import random
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)

def tanh(x):
  return math.tanh(x)

def tanh_derivative(x):
  return 1 - math.tanh(x) ** 2
  
class NeuralNet:
  def __init__(self, input_layer_neurons=4, hidden_layer_neurons=8, output_layer_neurons=3, percentage_to_train=80, function_activation='Sigmoid', csv_path=None, learning_rate=0.01, momentum=0.9):
    # Set number of layers
    self.input_layer_neurons = input_layer_neurons
    self.hidden_layer_neurons = hidden_layer_neurons
    self.output_layer_neurons = output_layer_neurons
    
    # Set the weights
    self.weights_input_hidden  = [[random.uniform(-1.0, 1.0) for _ in range(hidden_layer_neurons)] for _ in range(input_layer_neurons)]
    self.weights_hidden_output = [[random.uniform(-1.0, 1.0) for _ in range(output_layer_neurons)] for _ in range(hidden_layer_neurons)]
    self.bias_hidden           = [random.uniform(-1.0, 1.0) for _ in range(hidden_layer_neurons)]
    self.bias_output           = [random.uniform(-1.0, 1.0) for _ in range(output_layer_neurons)]
    
    # Initialize velocities for momentum
    self.vel_weights_input_hidden  = [[0.0 for _ in range(hidden_layer_neurons)] for _ in range(input_layer_neurons)]
    self.vel_weights_hidden_output = [[0.0 for _ in range(output_layer_neurons)] for _ in range(hidden_layer_neurons)]
    self.vel_bias_hidden           = [0.0 for _ in range(hidden_layer_neurons)]
    self.vel_bias_output           = [0.0 for _ in range(output_layer_neurons)]
    
    # Separate the data by use
    self.data_from_csv       = load_data_from_csv(csv_path)
    self.data_input          = self.data_from_csv[0]
    self.data_output         = self.data_from_csv[1]
    self.labels              = one_hot_encode(self.data_output)[1]
    self.percentage_to_train = int(len(self.data_input)*(percentage_to_train / 100))
    
    # Set the activation function
    self.activation_function   = sigmoid            if function_activation == 'Sigmoid' else tanh
    self.activation_derivative = sigmoid_derivative if function_activation == 'Sigmoid' else tanh_derivative
    
    self.learning_rate = learning_rate
    self.momentum      = momentum

  def forward_propagation(self, input_data):
    hidden_layer_input = []

    for j in range(self.hidden_layer_neurons):
      activation = self.bias_hidden[j]

      for i in range(self.input_layer_neurons):
        activation += input_data[i] * self.weights_input_hidden[i][j]
      hidden_layer_input.append(self.activation_function(activation))
    
    output_layer_input = []

    for k in range(self.output_layer_neurons):
      activation = self.bias_output[k]

      for j in range(self.hidden_layer_neurons):
        activation += hidden_layer_input[j] * self.weights_hidden_output[j][k]
      output_layer_input.append(self.activation_function(activation))
    
    return hidden_layer_input, output_layer_input
  
  def backward_propagation(self, input_data, hidden_layer_activation, output_layer_activation, expected_output):    
    error_output_layer = [expected_output[k] - output_layer_activation[k] for k in range(self.output_layer_neurons)]
    delta_output_layer = [error_output_layer[k] * self.activation_derivative(output_layer_activation[k]) for k in range(self.output_layer_neurons)]
    
    error_hidden_layer = [0.0] * self.hidden_layer_neurons

    for j in range(self.hidden_layer_neurons):
      for k in range(self.output_layer_neurons):
        error_hidden_layer[j] += delta_output_layer[k] * self.weights_hidden_output[j][k]
      error_hidden_layer[j] *= self.activation_derivative(hidden_layer_activation[j])
    
    for j in range(self.hidden_layer_neurons):
      for k in range(self.output_layer_neurons):
        self.vel_weights_hidden_output[j][k] = self.momentum * self.vel_weights_hidden_output[j][k] + self.learning_rate * hidden_layer_activation[j] * delta_output_layer[k]
        self.weights_hidden_output[j][k] += self.vel_weights_hidden_output[j][k]
    
    for k in range(self.output_layer_neurons):
      self.vel_bias_output[k] = self.momentum * self.vel_bias_output[k] + self.learning_rate * delta_output_layer[k]
      self.bias_output[k] += self.vel_bias_output[k]
    
    for i in range(self.input_layer_neurons):
      for j in range(self.hidden_layer_neurons):
        self.vel_weights_input_hidden[i][j] = self.momentum * self.vel_weights_input_hidden[i][j] + self.learning_rate * input_data[i] * error_hidden_layer[j]
        self.weights_input_hidden[i][j] += self.vel_weights_input_hidden[i][j]
    
    for j in range(self.hidden_layer_neurons):
      self.vel_bias_hidden[j] = self.momentum * self.vel_bias_hidden[j] + self.learning_rate * error_hidden_layer[j]
      self.bias_hidden[j] += self.vel_bias_hidden[j]

  # Data 0 => Inputs
  # Data 1 => Labels
  def train(self, epochs):
    expected_output, _ = one_hot_encode(self.data_output)

    for epoch in range(epochs):
      for data, expected in zip(self.data_input[:self.percentage_to_train], expected_output):
        hidden_layer_activation, output_layer_activation = self.forward_propagation(data)
        self.backward_propagation(data, hidden_layer_activation, output_layer_activation, expected)

      if epoch % 100 == 0:
        total_loss = 0

        for i in range(len(self.data_input[:self.percentage_to_train])):
          _, output_layer_activation = self.forward_propagation(self.data_input[i])

          for j in expected_output[i]:
            loss = (expected_output[i][j] - output_layer_activation[j]) ** 2
          total_loss += loss

        average_loss = total_loss / len(self.data_input[:self.percentage_to_train])
        print(f'Epoch {epoch}, Avg. Loss: {round(average_loss, 5)}, Total Loss: {round(total_loss, 5)}')
  
  
  def predict(self):
    data_to_predict  = self.data_input[self.percentage_to_train:]
    predict_expected = self.data_output[self.percentage_to_train:]

    with open('archive/IrisPredictMomemtumSigmoid.csv', 'w', newline='') as file:
      fieldnames = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species', 'PredictExpected']
      writer     = csv.DictWriter(file, fieldnames=fieldnames)
      writer.writeheader()

      for index, data in enumerate(data_to_predict, start=0):
        output_layer_activation = self.forward_propagation(data)[1]
        label_decode            = one_hot_decode([output_layer_activation], self.labels)[0]

        writer.writerow({'SepalLengthCm': data[0], 'SepalWidthCm': data[1], 'PetalLengthCm': data[2], 'PetalWidthCm': data[3], 'Species': label_decode, 'PredictExpected': predict_expected[index] })

    print('Execução encerrada!')


# Exemplo de uso
input_layer_neurons  = 4
hidden_layer_neurons = 8
output_layer_neurons = 3
percentage_to_train  = 75
function_activation  = 'Sigmoid'
csv_path             = 'archive/IrisShuffled.csv'

neural_net = NeuralNet(input_layer_neurons, hidden_layer_neurons, output_layer_neurons, percentage_to_train, function_activation, csv_path)
neural_net.train(epochs=5000)
neural_net.predict()
