from processing import *
from models import LogisticClassifier

data_dir = os.path.join(os.pardir, "data")

# generate array from json files
train_x, train_y, test_x, index = get_data_from_json(data_dir)

model = LogisticClassifier(epochs=150)
model.train(train_x, train_y)
test_y = model.test(test_x)
write_predictions_to_file(index, test_y, data_dir)
