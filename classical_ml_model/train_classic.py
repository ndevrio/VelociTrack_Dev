import sys
from time import time
from unittest import loader
from mmapdataset import ClassicDataset
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
import joblib


def main():
    ########################################
    ###  Process and featurize the data  ###
    ########################################
    #n_splits = 10
    #skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    #lst_accu_stratified = np.zeros((n_splits,))
    #cmat = np.zeros((n_splits, len(class_names), len(class_names)))
    #s = 0

    train_dataset = ClassicDataset(sys.argv[1], sys.argv[2], sys.argv[3], 0)
    test_dataset = ClassicDataset(sys.argv[1], sys.argv[2], sys.argv[3], 1)

    # print(len(train_dataset.data_tuples))
    # print(len(test_dataset.data_tuples))
    # print(train_dataset.data_tuples[0])

    loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    X, y = next(iter(loader))

    # Convert to NumPy
    X_np, y_np = X.numpy(), y.numpy()

    print(X_np.shape, y_np.shape)

    # 2. Train the model
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_np, y_np)

    log_name = "rf_model_" + sys.argv[2] + "_" + sys.argv[3] + ".joblib"

    # 3. Save the model to a file
    joblib.dump(rf, log_name)


if __name__ == "__main__":
    main()