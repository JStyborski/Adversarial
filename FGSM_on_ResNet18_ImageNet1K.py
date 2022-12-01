import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as tvtran

###############
# User Inputs #
###############

imgDir = 'Source_Images'
imgName = 'giant_panda.JPEG'

eps = 0.005

################
# Prepare Data #
################

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu') # Runs just as fast for only 1 image and small model

# Load the ImageNet labels JSON file and create related dictionaries
imageNetLabelsDict = json.load(open('imagenet_class_index.json'))
id2label = {int(k): v[1] for k, v in imageNetLabelsDict.items()}
label2id = {v: k for k, v in id2label.items()}

# Define the pytorch image transform to resize and convert to tensor
transform = tvtran.Compose([
    tvtran.Resize((224, 224)),
    tvtran.ToTensor()
])

# Import the original image and push to a pytorch tensor
imgPIL = Image.open(os.path.join(imgDir, imgName))
imgTens = transform(imgPIL).unsqueeze(0).to(device)
imgTens.requires_grad = True

# Prepare the truth label
trueLabel = imgName.split('.')[0]
trueIdx = label2id[trueLabel]
labelTens = torch.Tensor(1, 1000).fill_(0).to(device)
labelTens[0, trueIdx] = 1

################
# Define Model #
################

# Create model and set in eval mode
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.to(device)
model.eval()

# Define softmax function for later
softmax = torch.nn.Softmax(dim=1)

# Define loss function as CE
loss = torch.nn.CrossEntropyLoss()

######################
# Run Model Normally #
######################

# Get output for normal image
outputLogitsTens = model(imgTens)
outputProbsTens = softmax(outputLogitsTens)

# Get original probability of true class
trueProb = outputProbsTens[0, trueIdx].cpu().detach().tolist()

# Calculate loss and backprop
model.zero_grad()
cost = loss(outputProbsTens, labelTens).to(device)
cost.backward()

################
# Perform FGSM #
################

with torch.no_grad():

    # Using the loss gradient wrt the input image, calculate the adversarial image
    advImgTens = imgTens + eps * imgTens.grad.sign()
    advImgTens = torch.clamp(advImgTens, 0, 1)

    # Get output for the adversarial image
    advOutputLogitsTens = model(advImgTens)
    advOutputProbsTens = softmax(advOutputLogitsTens)

    # Get predictions
    predIdx = torch.topk(advOutputProbsTens, 1)[1].cpu().tolist()[0][0]
    predLabel = id2label[predIdx]
    predProb = advOutputProbsTens[0, predIdx].cpu().detach().tolist()
    advTrueProb = advOutputProbsTens[0, trueIdx].cpu().detach().tolist()

###############
# Plot Images #
###############


# From 4D tensor input, convert
def show_image(imgTens, titleText):

    # Convert tensor to array and transpose to HxWxC
    imgArr = imgTens.cpu().detach().numpy()
    imgArr = np.transpose(imgArr[0], (1, 2, 0))

    # Plot images
    plt.imshow(imgArr)
    plt.title(titleText)
    plt.show()


# Run the plotter function
show_image(imgTens, 'Orig Img: ' + str(trueIdx) + ' ' + trueLabel + ' ' + '{0:.4f}'.format(trueProb))
show_image(torch.clamp(imgTens.grad.sign(), 0, 1), 'Adversarial Perturbation, Epsilon = ' + str(eps))
show_image(advImgTens, 'Adv Img: ' + str(predIdx) + ' ' + predLabel + ' ' + '{0:.4f}'.format(predProb)
           + '\n' + 'Orig Class: ' + str(trueIdx) + ' ' + trueLabel + ' ' + '{0:.4f}'.format(advTrueProb))
