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

targetLabel = 'espresso'

eps = 0.005

################
# Prepare Data #
################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the ImageNet labels JSON file and create related dictionaries
imageNetLabelsDict = json.load(open('imagenet_class_index.json'))
id2label = {int(k): v[1] for k, v in imageNetLabelsDict.items()}
label2id = {v: k for k, v in id2label.items()}

# Prepare the truth label
trueLabel = imgName.split('.')[0]
trueIdx = label2id[trueLabel]

# Define the pytorch image transform to resize and convert to tensor
transform = tvtran.Compose([
    tvtran.Resize((224, 224)),
    tvtran.ToTensor()
])

# Import the original image and push to a pytorch tensor
imgPIL = Image.open(os.path.join(imgDir, imgName))
imgTens = transform(imgPIL).unsqueeze(0).to(device)

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

#########################
# Perform Targeted FGSM #
#########################

# Create label tensor using the target class as the "correct" class
targetIdx = label2id[targetLabel]
labelTens = torch.Tensor(1, 1000).fill_(0).to(device)
labelTens[0, targetIdx] = 1

# Initialize the adversarial input/output as the original input/output
advImgTens = imgTens.clone().detach().to(device)
advImgTens.requires_grad = True
advOutputLogitsTens = model(advImgTens)
advOutputProbsTens = softmax(advOutputLogitsTens)

# While the predicted class is not the target class, keep updating the adversarial image
iterCount = 0
while advOutputProbsTens.max(dim=1)[1].cpu().detach().tolist()[0] != targetIdx:

    # Update count
    iterCount += 1

    # Calculate loss and backprop
    model.zero_grad()
    cost = loss(advOutputProbsTens, labelTens).to(device)
    cost.backward()

    # Update adversarial image from previous loop using grad results from latest model run
    # This is gradient descent to minimize cost, but we've reframed the cost function to think that the correct class
    # is the adversarial target class. Instead of updating model weights, we update the image
    advImgTens = advImgTens - eps * advImgTens.grad.sign()
    #advImgTens = advImgTens - eps * advImgTens.grad \
    #             / torch.mean(torch.linalg.matrix_norm(advImgTens.grad)).cpu().detach().tolist()

    # Clamp input values
    advImgTens = torch.clamp(advImgTens, 0, 1)

    # Need to refresh input, otherwise the gradient tree of advImgTens keeps growing with every iteration
    advImgTens = advImgTens.clone().detach().to(device)
    advImgTens.requires_grad = True

    # Get output for adversarial image
    advOutputLogitsTens = model(advImgTens)
    advOutputProbsTens = softmax(advOutputLogitsTens)

print('Number of iterations to achieve target class: ' + str(iterCount))

# Get predictions
# By the while loop condition above, we know the output prediction is the targetIdx and targetLabel
predProb = advOutputProbsTens[0, targetIdx].cpu().detach().tolist()
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
show_image(torch.clamp((advImgTens - imgTens) / eps, 0, 1), 'Total Adversarial Perturbation, Epsilon = ' + str(eps))
show_image(advImgTens, 'Adv Img: ' + str(targetIdx) + ' ' + targetLabel + ' ' + '{0:.4f}'.format(predProb)
           + ' (' + str(iterCount) + ' Iter)'
           + '\n' + 'Orig Class: ' + str(trueIdx) + ' ' + trueLabel + ' ' + '{0:.4f}'.format(advTrueProb))
