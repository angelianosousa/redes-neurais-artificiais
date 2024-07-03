from models.NeuralNet import NeuralNet

input_layer_neurons  = 3
hidden_layer_neurons = 5
output_layer_neurons = 2

file_path  = 'archive/Iris.csv'
neural_net = NeuralNet(input_layer_neurons, hidden_layer_neurons, output_layer_neurons, file_path, 75)
data_to_predict = neural_net.data_input[0][neural_net.data_to_train:]
# print(neural_net.data_input[0][:neural_net.data_to_train])
# print(neural_net.data_to_train)
neural_net.train(10000)
neural_net.predict()
