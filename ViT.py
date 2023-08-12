import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models

from einops import rearrange
from tqdm import trange, tqdm


device = torch.device("mps")
torch.manual_seed(0)

batch_size = 64
num_epochs = 5
learn_rate = 1E-3
weight_decay = 1E-4

# Add Norm
transform_mnist = transforms.Compose([transforms.ToTensor()])

train_data = datasets.MNIST("data/", train=True, download=True, transform=transform_mnist)
test_data = datasets.MNIST("data/", train=False, download=True, transform=transforms.ToTensor())

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, hidden_size=256, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.Transformer(d_model=dim, num_encoder_layers=depth, nhead=heads, dim_feedforward=hidden_size)
        self.to_cls_token = nn.Identity()
        self.loss_fn = nn.CrossEntropyLoss()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        tgt = x
        x = self.transformer(x, tgt, src_mask=mask)
        
        #x = self.to_cls_token(x[:, 0])
        return tgt

    def train(self, dataloader, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader): 
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            pred = self.forward(X)
            print(pred)
            loss = self.loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


    def test(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)

                pred = self.forward(X)
                test_loss = self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n") 



def train(dataloader, model, optimizer, loss_fn):
    size = len(dataloader.dataset)
    for batch, (X, y) in tqdm(enumerate(dataloader)): 
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model.forward(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
    
    print(f"loss: {loss:>7f}")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model.forward(X)
            test_loss = loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n") 



# TODO: Fix this to look similar to VisionTransformer but with single channel input
#model = ViT(image_size=28, patch_size=7, num_classes=10, dim=64, 
#            depth=6, heads=8, hidden_size=256, channels=1).to(device)


# The built in VisionTransformer model only works with 3 channel images, for MINST must modify the channel input
model = models.VisionTransformer(image_size=28, patch_size=7, num_layers=6, num_heads=8, hidden_dim=784, 
                                 mlp_dim=1024, num_classes=10, dropout=0.2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
loss_fn = nn.CrossEntropyLoss()

for x in trange(num_epochs):
    train(train_dataloader, model, optimizer, loss_fn)


test(test_dataloader, model, loss_fn)
print("Done!")
