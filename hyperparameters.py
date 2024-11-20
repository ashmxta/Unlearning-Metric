import optuna
import torch
import torch.nn as nn
from train import train_fn  # Import your train_fn class
from model import lenet  # Import your lenet class
import os

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    epochs = trial.suggest_int('epochs', 10, 50)
    eps = trial.suggest_loguniform('eps', 0.1, 10)
    sigma = trial.suggest_uniform('sigma', 0.5, 1.5)
    
    # Create an instance of your train_fn class with the suggested hyperparameters
    model = train_fn(
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        dp=1,
        cn=1,
        eps=eps,
        sigma=sigma,
        architecture="lenet",
        dataset="MNIST",
        trainset=None,
        save_name=None,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    )
    
    for epoch in range(epochs):
        model.train(epoch)
        
    accuracy = model.validate()
    
    # Optuna minimizes the objective function, return the negative accuracy
    return -accuracy

# Create or load the existing study
study_name = 'lenet_study'
storage_name = 'sqlite:///optuna_study.db'
if not os.path.exists('optuna_study.db'):
    study = optuna.create_study(direction='minimize', storage=storage_name, study_name=study_name)
else:
    study = optuna.create_study(direction='minimize', storage=storage_name, study_name=study_name, load_if_exists=True)

# Optimize the objective function
study.optimize(objective, n_trials=100)

print('Best hyperparameters:', study.best_params)

# Re-train with the best hyperparameters
best_params = study.best_params
best_model = train_fn(
    lr=best_params['lr'],
    batch_size=best_params['batch_size'],
    epochs=best_params['epochs'],
    dp=1,
    cn=1,
    eps=best_params['eps'],
    sigma=best_params['sigma'],
    architecture="lenet",
    dataset="MNIST",
    trainset=None,
    save_name=None,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
)

for epoch in range(best_params['epochs']):
    best_model.train(epoch)

# Evaluate the model on the test set
test_accuracy = best_model.validate()
print(f'Test Accuracy with optimized hyperparameters: {test_accuracy * 100:.2f}%')
