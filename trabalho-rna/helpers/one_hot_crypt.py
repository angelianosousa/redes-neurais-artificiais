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
