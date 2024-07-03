import math
import random
import csv
import pdb

# Funções de ativação
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)

# Inicialização de pesos e biases
input_layer_neurons  = 3
hidden_layer_neurons = 5
output_layer_neurons = 2

weights_input_hidden  = [[random.uniform(-1.0, 1.0) for _ in range(hidden_layer_neurons)] for _ in range(input_layer_neurons)]
weights_hidden_output = [[random.uniform(-1.0, 1.0) for _ in range(output_layer_neurons)] for _ in range(hidden_layer_neurons)]
bias_hidden           = [random.uniform(-1.0, 1.0) for _ in range(hidden_layer_neurons)]
bias_output           = [random.uniform(-1.0, 1.0) for _ in range(output_layer_neurons)]

# Forward Propagation
def forward_propagation(input_data, input_layer_neurons, hidden_layer_neurons, output_layer_neurons):
  hidden_layer_input = []

  for j in range(hidden_layer_neurons):
    activation = bias_hidden[j]
    for i in range(input_layer_neurons):
      activation += input_data[i] * weights_input_hidden[i][j]
    hidden_layer_input.append(sigmoid(activation))
  
  output_layer_input = []

  for k in range(output_layer_neurons):
    activation = bias_output[k]
    for j in range(hidden_layer_neurons):
      activation += hidden_layer_input[j] * weights_hidden_output[j][k]
    output_layer_input.append(sigmoid(activation))
  
  return hidden_layer_input, output_layer_input

# Backpropagation
def backward_propagation(input_data, hidden_layer_activation, output_layer_activation, expected_output):
  global weights_input_hidden, weights_hidden_output, bias_hidden, bias_output
  
  error_output_layer = [expected_output[k] - output_layer_activation[k] for k in range(output_layer_neurons)]
  delta_output_layer = [error_output_layer[k] * sigmoid_derivative(output_layer_activation[k]) for k in range(output_layer_neurons)]
  
  error_hidden_layer = [0.0] * hidden_layer_neurons
  for j in range(hidden_layer_neurons):
    for k in range(output_layer_neurons):
      error_hidden_layer[j] += delta_output_layer[k] * weights_hidden_output[j][k]
    error_hidden_layer[j] *= sigmoid_derivative(hidden_layer_activation[j])
  
  for j in range(hidden_layer_neurons):
    for k in range(output_layer_neurons):
      weights_hidden_output[j][k] += hidden_layer_activation[j] * delta_output_layer[k]
  
  for k in range(output_layer_neurons):
    bias_output[k] += delta_output_layer[k]
  
  for i in range(input_layer_neurons):
    for j in range(hidden_layer_neurons):
      weights_input_hidden[i][j] += input_data[i] * error_hidden_layer[j]
  
  for j in range(hidden_layer_neurons):
    bias_hidden[j] += error_hidden_layer[j]

# Codificação One-Hot
def one_hot_encode(labels):
  unique_labels = list(set(labels))
  label_to_vec = {label: [1 if i == index else 0 for i in range(len(unique_labels))] for index, label in enumerate(unique_labels)}
  return [label_to_vec[label] for label in labels], label_to_vec

# Decodificação One-Hot
def one_hot_decode(encoded_labels, label_to_vec):
  vec_to_label = {tuple(v): k for k, v in label_to_vec.items()}
  
  decoded_labels = []
  for encoded in encoded_labels:
    closest_match = min(vec_to_label.keys(), key=lambda x: sum((a - b) ** 2 for a, b in zip(x, encoded)))
    decoded_labels.append(vec_to_label[closest_match])
  
  return decoded_labels

# Função para carregar dados de um arquivo CSV
def load_data_from_csv(filepath):
  data   = []
  labels = []
  with open(filepath, newline='') as csvfile:

    reader = csv.DictReader(csvfile)
    for row in reader:
      object = [float(row['SepalLengthCm']), float(row['SepalWidthCm']), float(row['PetalLengthCm']), float(row['PetalWidthCm'])]
      data.append(object)
      labels.append(row['Species'])
      
  return data, labels

# Treinamento
def train(input_data, string_labels, epochs):
  expected_output, label_to_vec = one_hot_encode(string_labels)

  for epoch in range(epochs):
    for data, expected in zip(input_data, expected_output):
      hidden_layer_activation, output_layer_activation = forward_propagation(data)
      backward_propagation(data, hidden_layer_activation, output_layer_activation, expected)

    if epoch % 100 == 0:
      total_loss = 0
      for i in range(len(input_data)):
        _, output_layer_activation = forward_propagation(input_data[i])

        for j in expected_output[i]:
          loss = (expected_output[i][j] - output_layer_activation[j]) ** 2
        total_loss += loss
      average_loss = total_loss / len(input_data)
      print(f'Epoch {epoch}, Loss: {average_loss}')

  return label_to_vec

# Previsão
def predict(input_data, label_to_vec):
  _, output_layer_activation = forward_propagation(input_data)
  return one_hot_decode([output_layer_activation], label_to_vec)

# # Exemplo de uso
# input_data, string_labels = load_data_from_csv('archive/Iris.csv')

# label_to_vec = train(input_data, string_labels, epochs=10000)

# new_data = [0.2, 0.85, 0.9, 0.65]
# prediction = predict(new_data, label_to_vec)
# print(f'Previsão para {new_data}: {prediction}')

def prompt():
  input_layer_neurons  = 3
  hidden_layer_neurons = 5
  output_layer_neurons = 2

  weights_input_hidden  = [[random.uniform(-1.0, 1.0) for _ in range(hidden_layer_neurons)] for _ in range(input_layer_neurons)]
  weights_hidden_output = [[random.uniform(-1.0, 1.0) for _ in range(output_layer_neurons)] for _ in range(hidden_layer_neurons)]
  bias_hidden           = [random.uniform(-1.0, 1.0) for _ in range(hidden_layer_neurons)]
  bias_output           = [random.uniform(-1.0, 1.0) for _ in range(output_layer_neurons)]