import os

idx = '.'

for directory, subdirectories, files in os.walk(idx):
    for file in files:
        print(os.path.join(directory, file))
