import pickle
from natsort import natsorted

split_file = "C:/Users/ellen/Documents/code/B-line_detection/intermediate/data/info/dataset_split_dictionary.pkl"
frame_file = "C:/Users/ellen/Documents/code/B-line_detection/intermediate/data/info/frames_dictionary.pkl"

with open(frame_file, 'rb') as f:
    data = pickle.load(f)


print(data["Case-112"])
print(natsorted(data))