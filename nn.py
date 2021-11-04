# %% Imports
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from torch.optim import SGD, Adam

# %% Dataset
# Transforms to add noise and augmentation
transformations = torchvision.transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307), ((0.3081))),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=3),
    ]
)

training_data = torchvision.datasets.MNIST(
    root=Path("nn_mnist", "data"),
    train=True,
    download=True,
    transform=transformations,
)

print(
    f"Training dataset shape: {training_data.data.shape} x {training_data.targets.shape}"
)

test_data = torchvision.datasets.MNIST(
    root=Path("nn_mnist", "data"),
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

print(f"Testing dataset shape: {test_data.data.shape} x {test_data.targets.shape}")


# Take first image in training data
training_data_img1 = training_data.data[0]

train_dataloader = torch.utils.data.DataLoader(
    training_data,
    batch_size=60000,
    shuffle=True,
    num_workers=0,
)

# Take batch and label from train dataloader
train_dataloader_batch = next(iter(train_dataloader))
train_dataloader_img1 = train_dataloader_batch[0][0].squeeze()
train_dataloader_label1 = train_dataloader_batch[1][0]

X_train = train_dataloader_batch[0].numpy()
y_train = train_dataloader_batch[1].numpy()

# Take batch and label from test dataloader
test_dataloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=10000,
    shuffle=False,
    num_workers=0,
)

test_dataloader_batch = next(iter(test_dataloader))
X_test = test_dataloader_batch[0].numpy()
y_test = test_dataloader_batch[1].numpy()

# %%
# Distribution of data
unique, counts = np.unique(y_train, return_counts=True)
dict(zip(unique, counts))

ax = sns.barplot(x=unique, y=counts, data=None)
ax.set_xlabel("Digits")
ax.set_ylabel("Frequency")

# %%
# Show before and after transforms
fig, ax = plt.subplots(1, 2)
ax[0].imshow(training_data_img1, cmap=mpl.cm.binary)
ax[0].title.set_text("original")
ax[0].axis("off")

ax[1].imshow(train_dataloader_img1, cmap=mpl.cm.binary)
ax[1].title.set_text("transformed")
ax[1].axis("off")

fig.suptitle("Sample training image")
plt.show()


# %%
# Neural network architecture
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28 * 1, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28 * 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# %%
# Training
net = NeuralNetClassifier(
    MLP,
    callbacks=[EarlyStopping()],
    criterion=nn.CrossEntropyLoss,
    max_epochs=50,
)

# Grid Search
parameters = [
    {
        "lr": [0.1, 0.001, 0.001],
        "optimizer": [SGD],
        "optimizer__momentum": [0.3, 0.6, 0.9],
    },
    {
        "lr": [0.1, 0.001, 0.001],
        "optimizer": [Adam],
    },
]

gs = GridSearchCV(
    net,
    parameters,
    refit=True,
    cv=5,
    return_train_score=True,
    verbose=3,
    scoring="accuracy",
    n_jobs=-1,
)

# Pass data as numpy
gs.fit(X_train, y_train)

# %%
# Save MLP gridsearch
joblib.dump(
    gs,
    Path(
        "nn_mnist",
        "results",
        "NN_gs.pkl",
    ),
)

# Save computed weights
gs.best_estimator_.save_params(
    f_params=Path(
        "nn_mnist",
        "results",
        "NN_weights.pkl",
    )
)

# Grid Search Parameter Table
grid_table = pd.concat(
    [
        pd.DataFrame(gs.cv_results_["params"]),
        pd.DataFrame(gs.cv_results_["mean_test_score"], columns=["Accuracy"]),
    ],
    axis=1,
)

grid_table.to_csv(Path("nn_mnist", "results", "NN_grid_table.csv"))

print(grid_table)
print(gs.best_params_)

# %% Testing
# Load GS

gs = joblib.load(
    Path(
        "nn_mnist",
        "results",
        "NN_gs.pkl",
    )
)

# Start best model and load weights
Net = gs.best_estimator_.initialize()
Net.load_params(
    Path(
        "nn_mnist",
        "results",
        "NN_weights.pkl",
    )
)
Net.check_is_fitted


# Pass data as numpy

y_pred = Net.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred,
)

conf_matrix = confusion_matrix(
    y_test,
    y_pred,
)

# %%
plt.matshow(
    conf_matrix,
    cmap=plt.cm.Blues,
)

plt.show()

print("MLP accuracy: " + str(accuracy))

# %% SVM training
SVM = SVC()

params = [{"kernel": ["rbf", "linear"], "C": [3, 6, 9]}]


# Grid Search for SVM
SVM_gs = GridSearchCV(
    SVM,
    param_grid=params,
    refit=True,
    cv=2,
    return_train_score=True,
    verbose=3,
    scoring="balanced_accuracy",
    n_jobs=-1,
)

# each image should become a vector
X_train_flat_test = X_train[:10000].squeeze().reshape((10000, 28 * 28))
y_train_test = y_train[:10000]
X_train_flat = X_train.squeeze().reshape((60000, 28 * 28))

SVM_gs.fit(X_train_flat_test, y_train_test)

# %%
# Save best SVM model
joblib.dump(
    SVM_gs,
    Path(
        "nn_mnist",
        "results",
        "SVM_gs.pkl",
    ),
)

# Grid Search Parameter Table

grid_table = pd.concat(
    [
        pd.DataFrame(SVM_gs.cv_results_["params"]),
        pd.DataFrame(SVM_gs.cv_results_["mean_test_score"], columns=["Accuracy"]),
    ],
    axis=1,
)

print(grid_table)
print(SVM_gs.best_params_)

# %% Testing
# Load GS
SVM_gs = joblib.load(
    Path(
        "nn_mnist",
        "results",
        "SVM_gs.pkl",
    ),
)


X_test_flat = X_test.squeeze().reshape((10000, 28 * 28))

y_pred1 = SVM_gs.best_estimator_.predict(X_test_flat)

accuracy1 = accuracy_score(
    y_test,
    y_pred1,
)

conf_matrix1 = confusion_matrix(
    y_test,
    y_pred1,
)

# %%
plt.matshow(
    conf_matrix1,
    cmap=plt.cm.Blues,
)

plt.show()

print("SVM accuracy: " + str(accuracy1))
# %%