from helpers.use_csv import load_data_from_csv
from helpers.one_hot_crypt import one_hot_encode, one_hot_decode

import random
import math

class NeuralNet:
  def __init__(self, input_layer_neurons, hidden_layer_neurons, output_layer_neurons, csv_path, percentage_to_train):
    self.input_layer_neurons   = input_layer_neurons
    self.hidden_layer_neurons  = hidden_layer_neurons
    self.output_layer_neurons  = output_layer_neurons
    self.weights_input_hidden  = [[random.uniform(-1.0, 1.0) for _ in range(hidden_layer_neurons)] for _ in range(input_layer_neurons)]
    self.weights_hidden_output = [[random.uniform(-1.0, 1.0) for _ in range(output_layer_neurons)] for _ in range(hidden_layer_neurons)]
    self.bias_hidden           = [random.uniform(-1.0, 1.0) for _ in range(hidden_layer_neurons)]
    self.bias_output           = [random.uniform(-1.0, 1.0) for _ in range(output_layer_neurons)]
    self.data_input            = load_data_from_csv(csv_path)
    self.data_to_train       = int(len(self.data_input[0])*(percentage_to_train/100))

  def sigmoid(self, x):
    return 1 / (1 + math.exp(-x))

  def sigmoid_derivative(self, x):
    return x * (1 - x)

  def forward_propagation(self, input_data):
    hidden_layer_input = []

    for j in range(self.hidden_layer_neurons):
      activation = self.bias_hidden[j]

      for i in range(self.input_layer_neurons):
        activation += input_data[i] * self.weights_input_hidden[i][j]
      hidden_layer_input.append(self.sigmoid(activation))
    
    output_layer_input = []

    for k in range(self.output_layer_neurons):
      activation = self.bias_output[k]
      for j in range(self.hidden_layer_neurons):
        activation += hidden_layer_input[j] * self.weights_hidden_output[j][k]
      output_layer_input.append(self.sigmoid(activation))
    
    return hidden_layer_input, output_layer_input
  
  def backward_propagation(self, input_data, hidden_layer_activation, output_layer_activation, expected_output):    
    error_output_layer = [expected_output[k] - output_layer_activation[k] for k in range(self.output_layer_neurons)]
    delta_output_layer = [error_output_layer[k] * self.sigmoid_derivative(output_layer_activation[k]) for k in range(self.output_layer_neurons)]
    
    error_hidden_layer = [0.0] * self.hidden_layer_neurons
    for j in range(self.hidden_layer_neurons):
      for k in range(self.output_layer_neurons):
        error_hidden_layer[j] += delta_output_layer[k] * self.weights_hidden_output[j][k]
      error_hidden_layer[j] *= self.sigmoid_derivative(hidden_layer_activation[j])
    
    for j in range(self.hidden_layer_neurons):
      for k in range(self.output_layer_neurons):
        self.weights_hidden_output[j][k] += hidden_layer_activation[j] * delta_output_layer[k]
    
    for k in range(self.output_layer_neurons):
      self.bias_output[k] += delta_output_layer[k]
    
    for i in range(self.input_layer_neurons):
      for j in range(self.hidden_layer_neurons):
        self.weights_input_hidden[i][j] += input_data[i] * error_hidden_layer[j]
    
    for j in range(self.hidden_layer_neurons):
      self.bias_hidden[j] += error_hidden_layer[j]

  # Data 0 => Inputs
  # Data 1 => Labels
  def train(self, epochs):
    expected_output, _ = one_hot_encode(self.data_input[1])

    for epoch in range(epochs):
      for data, expected in zip(self.data_input[0][:self.data_to_train], expected_output):
        hidden_layer_activation, output_layer_activation = self.forward_propagation(data)
        self.backward_propagation(data, hidden_layer_activation, output_layer_activation, expected)

      if epoch % 100 == 0:
        total_loss = 0

        for i in range(len(self.data_input[0][:self.data_to_train])):
          _, output_layer_activation = self.forward_propagation(self.data_input[0][i])

          for j in expected_output[i]:
            loss = (expected_output[i][j] - output_layer_activation[j]) ** 2
          total_loss += loss

        average_loss = total_loss / len(self.data_input[0][:self.data_to_train])
        print(f'Epoch {epoch}, Avg. Loss: {round(average_loss, 5)}, Total Loss: {round(total_loss, 5)}')

    return
  
  def predict(self):
    data_to_predict = self.data_input[0][self.data_to_train:]
    _, label_to_vec = one_hot_encode(self.data_input[1])

    for i in range(len(data_to_predict)):
      _, output_layer_activation = self.forward_propagation(data_to_predict[i])
      
      print(f'Previsão para: {data_to_predict[i]} é {one_hot_decode([output_layer_activation], label_to_vec)[0]}')
