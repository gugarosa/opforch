import opforch.math.general as g
import opforch.stream.parser as p
import opforch.stream.splitter as s
from opforch.models import KNNSupervisedOPF
from opforch.stream import loader

# Loading a .txt file to a tensor
txt = loader.load_txt("data/boat.txt")

# Parsing the loaded tensor
X, Y = p.parse_loader(txt)

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = s.split(X, Y, percentage=0.8, random_state=1)

# Splitting training into train and validation
X_train, X_val, Y_train, Y_val = s.split(
    X_train, Y_train, percentage=0.25, random_state=1
)

# Creates a KNNSupervisedOPF instance (use device='cuda' for GPU)
opf = KNNSupervisedOPF(max_k=10, distance="log_squared_euclidean", device="cpu")

# Fits training data with validation set for k-learning
opf.fit(X_train, Y_train, X_val, Y_val)

# Predicts new data
preds = opf.predict(X_test)

# Calculating accuracy
acc = g.opf_accuracy(Y_test, preds)

print(f"Accuracy: {acc}")
