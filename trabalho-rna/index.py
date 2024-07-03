from models.NeuralNet import NeuralNet

input_layer_neurons  = 3
hidden_layer_neurons = 5
output_layer_neurons = 2

file_path  = 'archive/Iris.csv'
neural_net = NeuralNet(input_layer_neurons, hidden_layer_neurons, output_layer_neurons, file_path)
label_to_vec = neural_net.train(10000)
print(label_to_vec)
# neural_net.predict()
# print(neural_net)
# neural_net.predict()

# print(f'Percentage ===> {neural_net.percentage}')