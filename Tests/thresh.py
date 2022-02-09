import cv2
import numpy as np
import torch
import torch
import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
print("")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.relu = nn.ReLU()
        
        self.pool = nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,kernel_size=(5,5), stride=(1, 1), padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,kernel_size=(5,5), stride=(1, 1), padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,kernel_size=(5,5), stride=(1, 1), padding=(0,0))
        self.linear1 = nn.Linear(120,84)
        self.linear2 = nn.Linear(84,10)
        
    def forward(self,x):

        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        
        x = x.reshape(x.shape[0],-1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

model = LeNet()
model.load_state_dict(torch.load("model.pth"))
model.eval()

if torch.cuda.is_available():
    model.cuda()

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def predict_image(img, model):
    xb = img.unsqueeze(0).to(device=device)
    yb = model(xb).to(device=device)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()


def find_success(lst):
    print("total found {}".format(str(lst.count(3))))
    
    print(" length of list {}".format(str(len(lst))))
    
    b=(lst.count(3)/len(lst))*100
    
    print("success ratio : {}".format(str(b)))
    

cap=cv2.VideoCapture(0)

print(cap)

list_success = []

frame_cnt = 0
while True:
    try:
        
        ret,frame = cap.read()
        print(ret)
        if ret:
            frame_cnt +=1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_yellow = np.array([18,49,49])
        up_yellow =  np.array([74,160,213])
        crop = frame



        mask = cv2.inRange(frame,low_yellow,up_yellow)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            c = max(contours,key=cv2.contourArea)

            M = cv2.moments(c)

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            x, y, w, h = cv2.boundingRect(c)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            crop = frame[x:x+w-50,y:y+h-50,:]

            crop = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            _,crop = cv2.threshold(crop,87,255,cv2.THRESH_BINARY_INV)
            crop = cv2.resize(crop,(32,32))
            img=torch.Tensor(crop)
            img = img.unsqueeze(0)


            a = predict_image(img,model)
            print(' predicted:', a)
            list_success.append(a)
        
   
        
        cv2.imshow("frame",frame)
    
        cv2.imshow("crop",crop)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    except:
        pass
    
    
    

    
 


    
    
    
cv2.destroyAllWindows()
cap.release()

find_success(list_success)

print("frame number {}".format(str(frame_cnt)))