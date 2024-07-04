from models.NeuralNet import NeuralNet

print(15*'-==-')
print('By default the algorithm start with the below config:')
print('Input layer neurons: 3')
print('Hidden layer neurons: 5')
print('Input layer neurons: 2')
print('Function activation: Sigmoid')
print('Data for training 75% and 25% to prediction')
change_config = str(input('Set a new config ? (y/n): '))
print(15*'-==-')

file_path  = 'archive/Iris.csv'

if change_config == 'y':
  input_layer_neurons   = int(input('Set number of input layer neurons: '))
  hidden_layer_neurons  = int(input('Set number of hidden layer neurons: '))
  output_layer_neurons  = int(input('Set number of output layer neurons: '))
  function_activation   = str(input('Function activation (s - Sigmoid | t - Tanh): '))
  function_activation   = 'Sigmoid' if function_activation == 's' else 'Tanh'
  percentage_data_train = int(input('Set a percentage to data train: '))
  neural_net            = NeuralNet(input_layer_neurons, hidden_layer_neurons, output_layer_neurons, percentage_to_train=percentage_data_train, function_activation=function_activation, csv_path=file_path)

else:
  neural_net = NeuralNet(percentage_to_train=75, csv_path=file_path)


neural_net.train(10000)
neural_net.predict()

# TOOD
# Adaptar para aceitar um numero arbitrário de camadas
# Adicionar termo momento e opção para usar ele ou não