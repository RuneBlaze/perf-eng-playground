from .data import FibDataset, FibDataPoint
from .models import ClassifierHead
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
import sys
sys.setrecursionlimit(20000)

def train(model: ClassifierHead, dataloader: DataLoader[FibDataPoint], num_epochs: int = 100) -> None:
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    losses = []
    for batch in tqdm(dataloader):
        x = batch['x']
        y = batch['y']
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        losses.append(loss.item())
        if len(losses) >= 1000:
            tqdm.write(f'Loss: {sum(losses) / len(losses)}')
            losses.clear()
        optimizer.step()
        

if __name__ == '__main__':
    dataloader = DataLoader(FibDataset(), batch_size=32, num_workers=4)
    model = ClassifierHead()
    train(model, dataloader)