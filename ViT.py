import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models

from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from tqdm import trange, tqdm


device = torch.device("mps")
torch.manual_seed(0)

# Hyperparameters
batch_size = 64
num_epochs = 5
learn_rate = 1E-3
weight_decay = 1E-4

# TODO: Add Norm to preprocessing for training set?
# Add greyscale to work with pytorch vision transformer, I think this made the model worse vs 1 channel
transform_mnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
    ])

# Basic MNIST dataset downloader and loader
train_data = datasets.MNIST("data/", train=True, download=True, transform=transform_mnist)
test_data = datasets.MNIST("data/", train=False, download=True, transform=transform_mnist)

# Cross validation
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        # Depth = number of transformer blocks
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_size=256, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        
        # Image height = image width
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Transform image --> embedding then normalize and linear
        # Add conv layer to transform 3 channels to dim
        # Rearrange to (batch, patch, channel), old rearrange for no conv2d
        self.patch_to_embedding = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=patch_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c (h) (w) -> b (h w) (c)'),
            #Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Apply transformer to embedding, specify parameters and optional dropout
        self.transformer = Transformer(dim, depth, heads, dim_head=64, mlp_dim=mlp_size, dropout=0.2)

        # Apply MLP to transformer output
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_size),
            nn.GELU(),
            nn.Linear(mlp_size, num_classes),
        )

    def forward(self, img, mask=None):
        x = self.patch_to_embedding(img)
        b, n, _ = x.shape

        # Reshape (1, 1, dim) --> (batch, 1, dim) and concatenate with embedding
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        # Apply Transformer
        x = self.transformer(x)

        # Remove class token for mlp
        x = x[:, 0]

        # Apply MLP
        return self.mlp_head(x)


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



# Because this model uses conv2d with 3 channel input we need to use a grayscale transform
#model = models.VisionTransformer(image_size=28, patch_size=7, num_layers=6, num_heads=8, hidden_dim=784, 
#                                 mlp_dim=1024, num_classes=10, dropout=0.2).to(device)


model = ViT(image_size=28, patch_size=7, num_classes=10, dim=784, depth=6, heads=8, mlp_size=1024, channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
loss_fn = nn.CrossEntropyLoss()

for x in trange(num_epochs):
    train(train_dataloader, model, optimizer, loss_fn)


test(test_dataloader, model, loss_fn)
print("Done!")
