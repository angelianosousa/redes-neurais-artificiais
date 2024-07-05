import csv
import random

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

def shuffle_csv(input_path, output_path):
  # Read the CSV file
  with open(input_path, 'r') as infile:
    reader = csv.reader(infile)
    rows = list(reader)
  
  # Separate header and data
  header = rows[0]
  data = rows[1:]
  
  # Shuffle the data
  random.shuffle(data)
  
  # Write the shuffled data to a new CSV file
  with open(output_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)
    writer.writerows(data)
