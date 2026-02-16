import opforch.math.general as g
import opforch.stream.parser as p
import opforch.stream.splitter as s
from opforch.models import SupervisedOPF
from opforch.stream import loader

# Loading a .txt file to a tensor
txt = loader.load_txt("data/boat.txt")

# Parsing the loaded tensor into features and labels
X, Y = p.parse_loader(txt)

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = s.split(X, Y, percentage=0.5, random_state=1)

# Creates a SupervisedOPF instance (use device='cuda' for GPU)
opf = SupervisedOPF(distance="log_squared_euclidean", device="cpu")

# Fits training data into the classifier
opf.fit(X_train, Y_train)

# Predicts new data
preds = opf.predict(X_test)

# Calculating accuracy
acc = g.opf_accuracy(Y_test, preds)

print(f"Accuracy: {acc}")
