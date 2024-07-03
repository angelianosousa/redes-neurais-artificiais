import csv

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