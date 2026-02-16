import opforch.math.general as g
import opforch.stream.parser as p
import opforch.stream.splitter as s
from opforch.models import UnsupervisedOPF
from opforch.stream import loader

# Loading a .txt file to a tensor
txt = loader.load_txt("data/boat.txt")

# Parsing the loaded tensor
X, Y = p.parse_loader(txt)

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = s.split(X, Y, percentage=0.5, random_state=1)

# Creates an UnsupervisedOPF instance (use device='cuda' for GPU)
opf = UnsupervisedOPF(
    min_k=1, max_k=10, distance="log_squared_euclidean", device="cpu"
)

# Fits training data for clustering
opf.fit(X_train, Y_train)

# Propagate labels from cluster roots if data is labeled
opf.propagate_labels()

# Predicts new data
preds, clusters = opf.predict(X_test)

# Calculating accuracy
acc = g.opf_accuracy(Y_test, preds)

print(f"Accuracy: {acc}")
