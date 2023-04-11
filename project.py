import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from utils import print_np_array_properties, imshow_batch, imshow_with_encoded_labels, encode_attrs, get_encoded_attrs, split_data
from tqdm import tqdm, trange
import warnings
import logging
# logger = logging.getLogger()
# logger.setLevel(logging.CRITICAL)
# warnings.filterwarnings("ignore")
# %matplotlib inline
torch.cuda.empty_cache()


attr_dir = './data/CUB_200_2011/attributes.txt'
class_dir = 'data/CUB_200_2011/CUB_200_2011/classes.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Available Device: {device}')


class CUB_200_2011(datasets.ImageFolder):
    """Custom dataset that includes image paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, img_id_file = 'data/CUB_200_2011/CUB_200_2011/images.txt',
                       attrs_label_file = 'data/CUB_200_2011/CUB_200_2011/attributes/image_attribute_labels.txt',
                       encoded_concat=True,
                       *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.encoded_concat=encoded_concat

        attr_set = dict()

        with open(attrs_label_file) as file:

            for idx, line in enumerate(file):
                img_id, attr_id= line.split(' ')[0], line.split(' ')[1]
                attr_value = 1 if int(line.split(' ')[2]) ==1 and int(line.split(' ')[3]) >= 3 else 0
                if img_id not in attr_set:
                    attr_set[img_id]= [attr_value]
                else:
                    attr_set[img_id].append(attr_value)

        x = [encoding for key, encoding in attr_set.items()]

        self.encoded_attrs = torch.tensor(x)

        img_id_dict = dict()
        with open(img_id_file) as file:
            for line in file:
                line=line.rstrip()
                img_id, img_label= line.split(' ')[0], line.split('/')[1]
                img_id_dict[img_label]= img_id

        self.img_id_dict = img_id_dict

    # override the __getitem__ method that dataloader calls
    def __getitem__(self, index):

        path, target = self.imgs[index]
        img = self.loader(path)
        img_label = path.split('/')[-1]
        img_id = self.img_id_dict[img_label]
        img_idx = int(img_id)-1

        attrs = self.encoded_attrs[img_idx]

        encoded_class = torch.zeros(200)
        encoded_class[target] = 1

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.encoded_concat:
            encoded_target = torch.cat((encoded_class, attrs), dim=0)

            return img, encoded_target

        else:
            return img, target, encoded_class, attrs


    def __len__(self):
        return len(self.imgs)




class CVAE(nn.Module):
    def __init__(self, latent_size = 128, num_classes = 512, frame_size=(224,224)):

        super(CVAE, self).__init__()

        self.mu=None
        self.logvar=None
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.frame_height, self.width = frame_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3 + self.num_classes, 64, kernel_size=3, padding='same'),
            # nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 32, kernel_size=3, padding='same'),
            # nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.mu_fc = nn.Linear(32*56*56, self.latent_size)

        self.logvar_fc = nn.Linear(32*56*56, self.latent_size)

        self.z_fc = nn.Linear(self.latent_size + self.num_classes, 32*56*56)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def encode(self, x, y):
        y = y.reshape(y.shape[0], self.num_classes, 1, 1).repeat(1, 1, self.frame_height, self.width)

        x = torch.cat([x,y], dim=1)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)

        return mu, logvar

    def reparam(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + eps * std
        return z


    def decode(self, z, y):
        # y = y.reshape(y.shape[0], self.num_classes, 1, 1).repeat(1, 1, 54, 44)
        # print_np_array_properties(y)

        z = torch.cat([z, y.view(y.size(0),-1)], dim=1)
        # print_np_array_properties(z)

        z = self.z_fc(z)

        z = z.view(z.size(0), 32, 56, 56)

        x = self.decoder(z)

        return x

    def forward(self, x, y):

        self.mu, self.logvar = self.encode(x,y)
        z = self.reparam(self.mu, self.logvar)
        x = self.decode(z, y)

        return x, self.mu, self.logvar

def cvae_loss(x_out, x, mu, logvar):

    BCE = F.binary_cross_entropy(x_out, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)




if __name__ == '__main__':

    num_workers = 0
    # how many samples per batch to load
    batch_size = 7

    with open(attr_dir) as file:
        attr_labels  = [line.rstrip().split(' ')[1] for line in file]

    with open(class_dir) as file:
        class_labels = [line.rstrip().split(' ')[1] for line in file]


    combined_target_labels = class_labels + attr_labels

    # print(len(combined_target_labels))


    ## Define preprocessing transformation applied to data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)) #scale images to 224x224
        ])


    data = CUB_200_2011(
        root='./data/CUB_200_2011/CUB_200_2011/images/',
        transform = transform,
        encoded_concat=True
    )


    train_data, valid_data, test_data= split_data(data)

    print(f'Training data size: {len(train_data)}')
    print(f'Validation data size: {len(valid_data)}')
    print(f'testing data size: {len(test_data)}')


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    # print(f'\nclass labels:\n {class_labels}')
    # print(f'\nattribute labels:\n {attr_labels}')


    dataiter = iter(train_loader)
    images, encoded_targets = next(dataiter)

    print_np_array_properties(encoded_targets[0])
    first_img_targets = get_encoded_attrs(combined_target_labels,encoded_targets[0])
    print(first_img_targets)
    imshow_batch(1, images)

    # imshow_with_encoded_labels(1, images, attrs, attr_labels)




    model = CVAE().to(device)
    model.apply(initialize_weights)

    print(model)

    # print_np_array_properties(images)

    x, mu, logvar = model(images.to(device), encoded_targets.to(device))


    print_np_array_properties(x, 'reconstructed x')
    print_np_array_properties(mu, 'mu')
    print_np_array_properties(logvar, 'logvar')


    loss_func = cvae_loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)



    n_epochs=10

    valid_loss_min = np.Inf

    for epoch in trange(1, n_epochs+1):

        train_loss=0.0
        valid_loss=0.0

        ## Training Step
        model.train()
        for images, labels in tqdm(train_loader):

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() # clar the gradients of all optimized variables
            x_out, mu, logvar = model(images, labels) # Forward pass

            loss = loss_func(x_out, images, mu, logvar) # calculate loss of the forward pass

            loss.backward() #calculate gradients based on the loss

            optimizer.step() # update weights

            train_loss += loss.item()*images.size(0)
        ## Validation Step

        model.eval()
        for images, labels in tqdm(valid_loader):
            images, labels = images.to(device), labels.to(device)

            x_out, mu, logvar = model(images, labels) # forward pass

            loss = loss_func(x_out, images, mu, logvar)

            valid_loss += loss.item() * images.size(0)

        # calculate Average loss
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

        #Print progress statement
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

        #Test after ach epoch
        # obtain one batch of test images
    #     dataiter = iter(test_loader)
    #     images, labels = next(dataiter)
    #     images = images[:10]
    #     labels = labels[:10]

    #     # get sample outputs
    #     output, _, _ = model(images.to(device), labels.to(device))
    #     # prep images for display
    #     images = images.numpy()

    #     # output is resized into a batch of iages
    #     output = output.to('cpu').view(10, 3, 218, 178)
    #     # use detach when it's an output that requires_grad
    #     output = output.detach().numpy()

    #     # plot the first ten input images and then reconstructed images
    #     fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

    #     # input images on top row, reconstructions on bottom
    #     for images, row in zip([images, output], axes):
    #         for img, ax in zip(images, row):
    #             ax.imshow(np.transpose(img, (1,2,0)))
    #             ax.get_xaxis().set_visible(False)
    #             ax.get_yaxis().set_visible(False)


        #Save model with the lowest validation loss
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), './data/CVAE_celebA.pt')
            valid_loss_min = valid_loss
