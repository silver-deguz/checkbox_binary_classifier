import torch
from torch import nn
from pathlib import Path 
from dataset import create_train_val_loaders
from checkbox_classifier import CheckboxClassifier
from evaluation import accuracy_metrics


def train(root, n_epochs, batch_size, learning_rate=0.003):
    # create the dataloaders
    train_loader, val_loader = create_train_val_loaders(root, batch_size)

    # instantiate model
    net = CheckboxClassifier()

    # set loss function and initialize optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    # training loop with validation
    # prints both training and validation losses and accuracies at each epoch
    # also prints the loss of every 20 minibatches
    for i in range(n_epochs):  
        train_loss = 0.0
        train_accy = 0.0

        net.train()
        for minibatch, (images, labels) in enumerate(train_loader):
            preds = net(images)
            preds = torch.squeeze(preds, -1)
            loss = criterion(preds, labels.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accy = accuracy_metrics(preds, labels.float())
            train_loss += loss.item()
            train_accy += accy

            if minibatch > 0 and minibatch % 20 == 0:
                print('Epoch %d Minibatch = %d Loss = %.2f '%(i, minibatch, loss.item()))

        avg_train_loss = train_loss / len(train_loader)
        avg_train_accy = train_accy / len(train_loader)

        print('******* Epoch %d | Average Training Loss = %.2f | Average Training Accuracy = %.2f '%(i, avg_train_loss, avg_train_accy))
        
        val_loss = 0.0 
        val_accy = 0.0

        net.eval()
        for minibatch, (images, labels) in enumerate(val_loader):
            with torch.no_grad():
                preds = net(images)
                preds = torch.squeeze(preds, -1)
                loss = criterion(preds, labels.float())
                accy = accuracy_metrics(preds, labels.float())

                val_loss += loss.item()
                val_accy += accy

        avg_val_loss = val_loss / len(val_loader)
        avg_val_accy = val_accy / len(val_loader)

        print('******* Epoch %d | Average Validation Loss = %.2f | Average Validation Accuracy = %.2f '%(i, avg_val_loss, avg_val_accy))
        
    # finished training, save the model
    out_dir = Path(f"{root}/weights")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net, f'{out_dir}/model_{n_epochs}.pt') 


if __name__ == "__main__":
    root = "/Users/jdeguzman/workspace/take-homes/roots_automation"
    n_epochs = 50 
    batch_size = 16
    train(root, n_epochs, batch_size)