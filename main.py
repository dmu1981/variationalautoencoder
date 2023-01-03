import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import torchvision
from tqdm import tqdm

device = "cuda"
LATENT_DIM = 4096
IMAGE_SIZE = 256
BATCH_SIZE = 32

class Down(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.relu = nn.GELU()
        self.fw = nn.Sequential(
          nn.Conv2d(in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=(3,3),
                    padding='same').to(device),
          nn.BatchNorm2d(num_features=mid_channels).to(device),
          nn.GELU(),
          nn.Conv2d(in_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=(3,3),
                    padding='same').to(device),
          nn.BatchNorm2d(num_features=out_channels).to(device),
          nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
        )

    def forward(self, x):
        return self.fw(x)

class Up(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.relu = nn.GELU()
        self.fw = nn.Sequential(
          nn.BatchNorm2d(num_features=in_channels).to(device),
          nn.GELU(),
          nn.Upsample(scale_factor=2, mode="bilinear",
                      align_corners=True),
          nn.Conv2d(in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=(3,3), padding='same').to(device),
          nn.BatchNorm2d(num_features=mid_channels).to(device),
          nn.GELU(),
          nn.Conv2d(in_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=(3,3), padding='same').to(device)
        )

    def forward(self, x):
        return self.fw(x)

class Perception(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()
        self.d1 = Down( 3,  8,  8)
        self.d2 = Down( 8, 16, 16)
        self.d3 = Down(16, 32, 32)
        self.d4 = Down(32, 64, 64)
        self.d5 = Down(64, 128, 128)
        self.d6 = Down(128, 256, 256)
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(4096, 512).to(device)
        self.dropout = nn.Dropout()
        self.linear2 = nn.Linear(512, 5).to(device)

    def forward(self, x):
        #x = self.encoder(x)
        x = self.d1(x)
        emb1 = self.flatten(x)
        x = self.d2(self.gelu(x))
        emb2 = self.flatten(x) / 2
        x = self.d3(self.gelu(x))
        emb3 = self.flatten(x) / 3
        x = self.d4(self.gelu(x))
        emb4 = self.flatten(x) / 4
        x = self.d5(self.gelu(x))
        emb5 = self.flatten(x) / 5
        x = self.d6(self.gelu(x))
        emb6 = self.flatten(x) / 6

        x = self.gelu(self.linear1(self.flatten(self.gelu(x))))
        x = self.linear2(self.dropout(x))

        emb = torch.cat([emb1, emb2, emb3, emb4, emb5, emb6],dim=1)
        return F.log_softmax(x, dim=1), emb

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
          Down( 3,  8,  8), nn.GELU(),
          Down( 8, 16, 16), nn.GELU(),
          Down(16, 32, 32), nn.GELU(),
          Down(32, 64, 64), nn.GELU(),
          Down(64, 128, 128), nn.GELU(),
          Down(128, 256, 256), nn.GELU(),
          nn.Flatten(),
        )
        self.mu = nn.Linear(4096, LATENT_DIM)
        self.std = nn.Linear(4096, LATENT_DIM)

    def forward(self, x):
        x = self.encoder(x)
        return self.mu(x), torch.exp(self.std(x))

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(LATENT_DIM, 4096)
        self.gelu = nn.GELU()
        self.decoder = nn.Sequential(
          Up(256, 256, 128),
          Up(128, 128, 64),
          Up(64, 64, 32),
          Up(32, 32, 16),
          Up(16, 16, 16),
          Up(16, 16, 3),
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.gelu(self.linear(x))
        return self.tanh(self.decoder(x.view(-1,256,4,4)))

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dist = torch.distributions.Normal(0, 1)
        self.dist.loc = self.dist.loc.cuda()
        self.dist.scale = self.dist.scale.cuda()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, std = self.encoder(x)
        z = mu + std * self.dist.sample(mu.shape)
        x = self.decoder(z)
        self.kl = (std**2 + mu**2 - torch.log(std) - 1).mean()
        return x, z

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), 'replicate')
        x_diff = x[..., :-1, :-1] - x[..., :-1, 1:]
        y_diff = x[..., :-1, :-1] - x[..., 1:, :-1]
        diff = x_diff**2 + y_diff**2
        return diff.mean()

def get_data(dataset_path, batch_size):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        torchvision.transforms.Lambda(lambda x: x.to(device))
    ])
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


dataloader = get_data("./flowers", BATCH_SIZE)
PERCEPTION_PATH = "perception.pt"
VAE_PATH = "vae.pt"

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    im = torchvision.transforms.ToPILImage()(grid)
    im.save(path)

def train_perception():
    model = Perception()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_function = nn.NLLLoss()

    for epoch in range(25):
        total_loss = 0
        cnt = 0
        print("epoch: ", epoch)
        correct = 0
        best_acc = 0
        for batch, labels in dataloader:
            #batch = batch.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            x, emb = model(batch)

            correct += torch.sum((torch.argmax(x, dim=1) == labels.view(-1)).float())

            loss = loss_function(x, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            cnt += 1

            acc = correct / (cnt * BATCH_SIZE) * 100
            if cnt % 10 == 0:
                print('loss {:.8f}, acc {:.2f}%%'.
                    format(total_loss / cnt,
                           correct / (cnt * BATCH_SIZE) * 100))

        if acc > best_acc:
            print("Saving model with best accuracy so far")
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, PERCEPTION_PATH)
            best_acc = acc

def eval_vae():
    model = VAE()
    model.to(device)

    try:
        checkpoint = torch.load(VAE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        print("Could not load model from disk")
        exit()

    batch, _ = next(iter(dataloader))
    img, emb = model(batch)
    z1 = emb[0]
    z2 = emb[1]

    z = torch.Tensor(size=(BATCH_SIZE, LATENT_DIM))
    for i in range(BATCH_SIZE):
        p = i / (BATCH_SIZE - 1)
        z[i,:] = z1 * (1-p) + z2 * p

    img = model.decoder(z.cuda())
    img = (img + 1) / 2
    save_images(img, "interpolated.png")


def train_vae():
    perception = Perception()
    perception.to(device)
    checkpoint = torch.load(PERCEPTION_PATH)
    perception.load_state_dict(checkpoint['model_state_dict'])
    perception.eval()

    model = VAE()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    tvloss = TotalVariationLoss()

    try:
        checkpoint = torch.load(VAE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        print("Could not load model from disk, starting from scratch")

    loss_function = nn.L1Loss()
    skip_save = 5
    best_loss = 999999
    for epoch in range(2500):
        total_loss = 0
        cnt = 0

        print("epoch: ", epoch)
        bar = tqdm(dataloader)
        for batch, _ in bar:
            with torch.no_grad():
                _, gt_emb = perception(batch)

            optimizer.zero_grad()
            x, _ = model(batch)

            _, mdl_emb = perception(x)

            loss = 5 * loss_function(x, batch) +\
                       loss_function(mdl_emb, gt_emb) +\
                 0.5 * tvloss(x) +\
                       model.kl
         
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            cnt += 1
            bar.set_description("{:.8f}".format(total_loss / cnt))

        batch, _ = next(iter(dataloader))
        img, _ = model(batch)
        img = (img + 1) / 2
        save_images(img, "epoch_{}.png".format(epoch))

        total_loss /= cnt
        print("final loss: {:.8f}".format(total_loss))
        if skip_save > 0:
            skip_save -= 1

        if total_loss < best_loss:
            best_loss = total_loss
            if skip_save <= 0:
                print("Saving model with best loss so far")
                torch.save({
                          'epoch': epoch,
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          }, VAE_PATH)
                skip_save = 5
            else:
                print("Would save but skipping,", skip_save, "epochs to go")

#train_perception()
#train_vae(True)
eval_vae()
