import torch
from model import *
import argparse
from torch.utils.data import Dataset, DataLoader
import pandas as pd


def parse_args(node_id,mode):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="path to model", type=str, default="./checkpoints/{}_epoc.pt".format(node_id))
    parser.add_argument('-i', '--input', help="path to input files", type=str, default="./data/v1/test_{}.csv".format(mode))
    parser.add_argument('-o', '--output', help="path to output files", type=str, default="output_{}.txt".format(mode))
    parser.add_argument('--input-features', help="input dimension for model", type=int, default=30)
    parser.add_argument('--output-features', help="output dimension for model", type=int, default=2)

    return parser.parse_args()


class SeedDataset(Dataset):

    def __init__(self, annotations_file):
        super().__init__()
        self.data: pd.DataFrame = pd.read_csv(annotations_file)
        self.X = self.data.drop(columns=['id']).fillna(value=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.as_tensor(self.X.iloc[idx].values).type(torch.FloatTensor)


def output(node_id,outputfile):
    args = parse_args(node_id,outputfile)
    model = FCModel(args.input_features, args.output_features)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    test_dataset = SeedDataset(args.input)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    outputs = []
    for x in test_dataloader:
        logit = model(x)
        outputs.append(str(logit.argmax(1).item()))

    with open(args.output, 'w') as f:
        f.write('\n'.join(outputs))
    print("output_{}".format(outputfile))


if __name__ == "__main__":
    output(12,'b')
