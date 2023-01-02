import torch 
from sklearn.metrics import roc_auc_score, accuracy_score
from dataset import create_val_loader, create_test_loader


def accuracy_metrics(preds, targets):
    preds_label = torch.round(torch.sigmoid(preds)).detach().numpy()
    targets_label = targets.detach().numpy()
    accy = accuracy_score(targets_label, preds_label) * 100
    return accy


def auc_metrics(preds, targets):
    preds_prob = torch.sigmoid(preds).detach().numpy()
    targets_label = targets.detach().numpy()
    auc = roc_auc_score(targets_label, preds_prob)
    return auc 


def predict_and_evaluate(input_dir, weights_path):
    # create val/test dataloader
    data_loader = create_test_loader(input_dir)

    # load model weights and set to eval
    net = torch.load(weights_path)
    net.eval()
    
    preds = torch.ones(len(data_loader), dtype=torch.float32)
    targets = torch.ones(len(data_loader), dtype=torch.int64)

    # accumulate all predictions and targets into a list
    for i, (images, labels) in enumerate(data_loader):
        with torch.no_grad():
            p = net(images)
            p = torch.squeeze(p, -1)
        preds[i] = p
        targets[i] = labels.float()
    
    # calculate accuracy and auc metrics on full predictions
    accy = accuracy_metrics(preds, targets)
    auc = auc_metrics(preds, targets)
    
    print('Accuracy = %.2f | AUC = %.2f '%(accy, auc))

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help="/path/to/directory/of/images")
    parser.add_argument("--weights", required=True, type=str, help="/path/to/pretrained/model/weights")
    args = parser.parse_args()

    predict_and_evaluate(args.input, args.weights)
