import numpy as np

import torch.nn as nn
import argparse
import os
import random
import shutil
import time
import warnings

from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
from PIL import Image
from data import data_transforms , data_transforms_aug
from utils import *
import math
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import matplotlib.patches as mpatches



#FOR DATA GENERATION

def count_pixels_from_each_class(mask):

    shape = mask.shape

    count255 = 0
    count0 = 0


    for i in range(0,shape[0]):
        for j in range(0,shape[1]):
            if mask[i][j] == 255:
                count255 += 1
            if mask[i][j] == 0:
                count0 += 1

    return(count0,count255)



def is_in_the_neighbourhood_of_crack_pixel(center_of_crop,mask,authorized_distance):
    bool = False
    d = int(authorized_distance)
    a = int(center_of_crop[0]) - d
    b = int(center_of_crop[1]) - d
    for i in range(a , a + 2*d):
        for j in range(b , b + 2*d):
            if mask[i][j] == 0:
                bool = True
                return(bool)
    return(bool)


def intersection(x , y , size_of_images , crop):
    rect1_right = x + size_of_images/2
    rect1_left = x - size_of_images/2
    rect1_top = y - size_of_images/2
    rect1_bottom = y + size_of_images/2
    rect2_right = crop[1] + size_of_images/2
    rect2_left = crop[1] - size_of_images/2
    rect2_top = crop[2] - size_of_images/2
    rect2_bottom = crop[2] + size_of_images/2
    x_overlap = np.maximum(0, np.minimum(rect1_right, rect2_right) - np.maximum(rect1_left, rect2_left));
    y_overlap = np.maximum(0, np.minimum(rect1_bottom, rect2_bottom) - np.maximum(rect1_top, rect2_top));
    overlapArea = x_overlap * y_overlap;
    return(overlapArea/(size_of_images)**2)


#size_of_images and nb_data must be even numbers

def generate_data_from_one_image(image,mask,size_of_images,distance_from_centroid,authorized_overlap):
    print('generating....')
    b = mask.shape[0]
    a = mask.shape[1]
    print(mask.shape)
    print(image.size)

#    assert image.size[0] >= size_of_images
#    assert image.size[0] == mask.shape[0]
#    assert image.size[1] == mask.shape[1]

    image_array = np.array(image)

    Positive_crops = []
    Negative_crops = []
    Negative_potential_positions = []
    c=0
    #count0 , count255 = count_pixels_from_each_class(mask)
    count0 , count255 = 0.995 , 0.005
#    proportion_of_crack_pixels = count255/(count0+count255)
#    print(proportion_of_crack_pixels)
    temp_positive_data = 0
    temp_negative_data = 0

    step = 2*distance_from_centroid -1
    x , y = size_of_images/2 , size_of_images/2
    number_of_steps = int(a/(step))
#    print(number_of_steps)
    positive_temporary = []


    for i in range (0,number_of_steps-1):
        c += 1
#        print("step number %s / %s "%(c , number_of_steps))
        l = len(Negative_potential_positions)
#        print("length of the temporary list containing potential negative positions : %s  "%(l))
        while x < b - size_of_images/2 and y < a - size_of_images/2:
            center_of_crop = [x,y]
            if is_in_the_neighbourhood_of_crack_pixel(center_of_crop , mask , distance_from_centroid):
                to_be_added = True
                for crop in positive_temporary:
                    if intersection(x , y , size_of_images , crop) > authorized_overlap:
                        to_be_added = False
                        break
                if to_be_added == True:
                    cropped_image = image.crop((y-size_of_images/2 , x-size_of_images/2 , y+size_of_images/2 , x+size_of_images/2))
#                cropped_image = image.crop((x-size_of_images/2 , y-size_of_images/2 , x+size_of_images/2 , y+size_of_images/2))
                    Positive_crops.append([cropped_image,x,y])
                    positive_temporary.append([cropped_image,x,y])
                    temp_positive_data += 1
                    if temp_positive_data%10 ==0:
                        print("Positive data : %s"%(temp_positive_data))
                    y += step
                else:
                    y += step
            else:
                p = random.random()
                if p < 0.1:
                    Negative_potential_positions.append(center_of_crop)
                y += step
#            for pos in Negative_potential_positions:
#                if pos[0] <= x - size_of_images:
#                    to_be_added = True
#                    for crop in positive_temporary:
#                        if intersection(pos[0] , pos[1] , size_of_images , crop) > 0.0001:
#                            to_be_added = False
#                            break
#                    if to_be_added == True:
#                        cropped_image = image.crop((pos[1]-size_of_images/2 , pos[0]-size_of_images/2 , pos[1]+size_of_images/2 , #pos[0]+size_of_images/2))
#                        Negative_crops.append([cropped_image,pos[0],pos[1]])
#                        temp_negative_data +=1
#                        if temp_negative_data%10 ==0:
#                            print("Negative data : %s"%(temp_negative_data))
#                        Negative_potential_positions.remove(pos)
#                    else:
#                        Negative_potential_positions.remove(pos)
            for crop in positive_temporary:
                if crop[1] < (x-3*size_of_images):
                    positive_temporary.remove(crop)

        x += step
        y = size_of_images/2
    for pos in Negative_potential_positions:
        to_be_added = True
        for crop in Positive_crops:
            if intersection(pos[0] , pos[1] , size_of_images , crop) > 0.0001:
                to_be_added = False
#                break
        if to_be_added == True:
            cropped_image = image.crop((pos[1]-size_of_images/2 , pos[0]-size_of_images/2 , pos[1]+size_of_images/2 , pos[0]+size_of_images/2))
            Negative_crops.append([cropped_image,pos[0],pos[1]])
            temp_negative_data +=1
            if temp_negative_data%10 ==0:
                print("Negative data : %s"%(temp_negative_data))
            Negative_potential_positions.remove(pos)
        else:
            Negative_potential_positions.remove(pos)
                        
    print(len(Positive_crops))
    print(len(Negative_crops))
    print('done')
    return(Positive_crops,Negative_crops)



#From the path, make sure that there is one folder called "Masks", and another called Images to load the data, and
#In this script, I create the Positive and Negative folders, which can be modified to just add the files to an existing folder
def generate_data(path , size_of_images , distance_from_centroid , authorized_overlap):
    l = len(os.listdir(path + '/delft_mask'))
    c = 0
    for file in os.listdir(path + '/delft_mask'):
        c += 1
        print("%s / %s " %(c,l))
        mask_name = file
        image_name = file[:-9] + '.jpg'
        print(file)
#        if image_name[:-4] +'_0' + '.jpg' in os.listdir(path + '/new_data/new_positive2'):
#            print(file)
#            continue
#        if image_name[0:4] == '2019':
#            print(file)
#            continue
        
        image_temp = Image.open(path + '/images_delft/' + image_name)
        mask = Image.open(path + '/delft_mask/' + mask_name)
        matrix = np.array(mask)
        if image_temp.size[0] == matrix.shape[1]:
            image = image_temp
        else:
            image = image_temp.rotate(270 , expand = True)        
        pos, neg = generate_data_from_one_image(image , matrix , size_of_images , distance_from_centroid , authorized_overlap)


        #So that the image generates has many Positive crops than Negative crops
        p = len(pos)
        n = len(neg)
        a = n - p
        
        if n > p:

            indexes_to_remove = random.sample(range(0 , n) , a)
            indexes_to_remove.sort(reverse = True)
            for index in indexes_to_remove:
                neg.remove(neg[index])
        else:
            print('Positive data : %s , Negative data : %s' %(p,n)) 
        count_pos = 0
        for crop in pos:
            crop[0].save('/home/eve/crack_detection/delft_cracks/' +image_name[:-4] + '_' + str(count_pos) +  '.jpg' , 'JPEG')
            count_pos += 1
        count_neg = 0
#        for crop in neg:
#            crop[0].save('/home/eve/crack_detection/new_data/' + 'new_negative2/' + image_name[:-4] + '_' + str(count_neg) +  '.jpg' , 'JPEG')
#            count_neg += 1
            
def generate_data_spp(path , dict_of_images ,  distance_from_centroid , authorized_overlap):
    
    List_remove_neg = ['200190703_095711' , '200190703_100029' , '200190703_101856' , '200190703_102226' , '200190703_103007' , '200190703_103730' , '200190703_104558' , '20190705_124518' , '20190705_143048' , '20190705_143342']
    
    for key , L in dict_of_images.items():
        
        Folder = '/home/eve/crack_detection/spp_data/' + str(key) + '_crops'
        
        for file in L:  
            print(file)
            print(key)
            mask_name = file + '_mask.tif'
            image_name = file + '.jpg'
            listfiles = os.listdir('/home/eve/crack_detection/spp_data/180_crops/positive') + os.listdir('/home/eve/crack_detection/spp_data/227_crops/positive') + os.listdir('/home/eve/crack_detection/spp_data/350_crops/positive') + os.listdir('/home/eve/crack_detection/spp_data/470_crops/positive')
            test_name = file + '_0.jpg'
            if test_name in listfiles:
                print('already generated')
                continue
            image = Image.open(path + '/Images/' + image_name)
            mask = Image.open(path + '/Masks/' + mask_name)
            matrix = np.array(mask)
            pos, neg = generate_data_from_one_image(image , matrix , key , distance_from_centroid , authorized_overlap)
            if file in List_remove_neg:
                count_pos = 0
                for crop in pos:
                    crop[0].save(Folder + '/positive/' + file + '_' + str(count_pos) +  '.jpg' , 'JPEG')
                    count_pos += 1
                print('Positive data : %s :'  %(count_pos))
            else:
                p = len(pos)
                n = len(neg)
                a = n - p
                if n > p:

                    indexes_to_remove = random.sample(range(0 , n) , a)
                    indexes_to_remove.sort(reverse = True)
                    for index in indexes_to_remove:
                        neg.remove(neg[index])
                else:
                    print('Positive data : %s , Negative data : %s' %(p,n)) 
                count_pos = 0
                for crop in pos:
                    crop[0].save(Folder + '/positive/' + file + '_' + str(count_pos) +  '.jpg' , 'JPEG')
                    count_pos += 1
                count_neg = 0
                for crop in neg:
                    crop[0].save(Folder + '/negative/' + file + '_' + str(count_neg) +  '.jpg' , 'JPEG')
                    count_neg += 1
                print('next image')
            
            
            
# FOR TRAINING THE MODEL


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    
    return model

def train(epoch , model , train_loader , optimizer , args , use_cuda):
    Loss_values = []
    model.train()
    torch.cuda.empty_cache()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        #set_trace()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
       
        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            Loss_values.append(loss.data.item())

def train_spp(model , list_paths):
    
    L = []
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005, amsgrad=False)
    for epoch in range(15):
        for data_path in list_paths:
            validation_split = 0.1
            dataset_size = len(os.listdir(data_path + '/positive')) + len(os.listdir(data_path + '/negative'))
            valid_length = int(dataset_size * validation_split)
            train_length = dataset_size - valid_length
            dataset = datasets.ImageFolder(data_path,transform=data_transforms_global)

            trainset , validset = torch.utils.data.random_split(dataset , (train_length , valid_length))

            train_loader = torch.utils.data.DataLoader(trainset,
                    batch_size=args['batch_size'],
                    num_workers=4)

            valid_loader = torch.utils.data.DataLoader(validset,
                    batch_size=args['batch_size'],
                    num_workers=4)

            Loss_values = []
            train(epoch,model)
            L.extend(Loss_values)
            print('Training done')

            
def validation(model , valid_loader):
    with torch.no_grad():
        model.eval()
        torch.cuda.empty_cache()
        validation_loss = 0
        correct = 0
        for data, target in valid_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            validation_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        validation_loss /= len(valid_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validation_loss, correct, len(valid_loader.dataset),
            100. * correct / len(valid_loader.dataset)))
        
        
def output_space():
    model.eval()
    outputspace = np.zeros((num_classes,n_test,num_classes))
    j=0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = F.softmax(model(data), dim=1).detach().numpy()
        outputspace[j] = output
        j += 1
    return outputspace

def adjust_learning_rate_adam(optimizer, epoch):
    initial_lr = [0.02,0.02,0.02,0.02]
    """Sets the learning rate to the initial LR decayed by 2 every 2 epochs"""
#    optimizer.param_groups[0]['lr'] = initial_lr[0]*(0.1**(epoch // 5))
#    optimizer.param_groups[1]['lr'] = initial_lr[1]*(0.1**(epoch // 5))
#    optimizer.param_groups[2]['lr'] = initial_lr[2]*(0.1**(epoch // 5))
#    optimizer.param_groups[3]['lr'] = initial_lr[3]*(0.1**(epoch // 5))
    if epoch == 1 :
        optimizer.param_groups[0]['lr'] = 10**(-4)
    if epoch == 3:
        optimizer.param_groups[0]['lr'] = 10**(-5)
    if epoch == 6 :
        optimizer.param_groups[0]['lr'] = 10**(-6)
    if epoch == 10 :
        optimizer.param_groups[0]['lr'] = 10**(-7)
                


def adjust_learning_rate(optimizer, epoch):
    initial_lr = [0.02,0.02,0.02,0.02]
    """Sets the learning rate to the initial LR decayed by 2 every 2 epochs"""
#    optimizer.param_groups[0]['lr'] = initial_lr[0]*(0.1**(epoch // 5))
#    optimizer.param_groups[1]['lr'] = initial_lr[1]*(0.1**(epoch // 5))
#    optimizer.param_groups[2]['lr'] = initial_lr[2]*(0.1**(epoch // 5))
#    optimizer.param_groups[3]['lr'] = initial_lr[3]*(0.1**(epoch // 5))
    if epoch == 1 :
        optimizer.param_groups[0]['lr'] = 10**(-2)
        optimizer.param_groups[1]['lr'] = 10**(-2)
        optimizer.param_groups[2]['lr'] = 10**(-2)
        optimizer.param_groups[3]['lr'] = 10**(-2)
    if epoch == 5:
        optimizer.param_groups[0]['lr'] = 5*10**(-3)
        optimizer.param_groups[1]['lr'] = 5*10**(-3)
        optimizer.param_groups[2]['lr'] = 5*10**(-3)
        optimizer.param_groups[3]['lr'] = 5*10**(-3)
    if epoch == 8 :
        optimizer.param_groups[0]['lr'] = 5*10**(-4)
        optimizer.param_groups[1]['lr'] = 5*10**(-4)
        optimizer.param_groups[2]['lr'] = 5*10**(-4)
        optimizer.param_groups[3]['lr'] = 5*10**(-4)
    if epoch == 11 :
        optimizer.param_groups[0]['lr'] = 5*10**(-5)
        optimizer.param_groups[1]['lr'] = 5*10**(-5)
        optimizer.param_groups[2]['lr'] = 5*10**(-5)
        optimizer.param_groups[3]['lr'] = 5*10**(-5)
    if epoch == 12 :
        optimizer.param_groups[0]['lr'] = 10**(-5)
        optimizer.param_groups[1]['lr'] = 10**(-5)
        optimizer.param_groups[2]['lr'] = 10**(-5)
        optimizer.param_groups[3]['lr'] = 10**(-5)
    if epoch == 16 :
        optimizer.param_groups[0]['lr'] = 10**(-6)
        optimizer.param_groups[1]['lr'] = 10**(-6)
        optimizer.param_groups[2]['lr'] = 10**(-6)
        optimizer.param_groups[3]['lr'] = 10**(-6)
        
#FOR TESTING THE MODEL


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


#Go through the whole image using a sliding windows and classifying each crop. The function returns the list of crops, with their positions and probability (confidence) score.


def sliding_window(image , model , overlapping_percentage , size_of_images):
    c1 , c2 = 0 , 0
    a , b = image.size
    number_of_steps = int(b/(size_of_images*(1-overlapping_percentage)))
    x , y = int(size_of_images/2) + 1 , int(size_of_images/2) + 1
    L = []
    for i in range (0,number_of_steps-1):
        y = int(size_of_images/2) + 1
        while x < b - size_of_images/2 and y < a - size_of_images/2:
            c1 += 1
            cropped_image = image.crop((y-size_of_images/2 , x-size_of_images/2 , y+size_of_images/2 , x+size_of_images/2)) 
            crop = data_transforms(cropped_image)
            crop = crop.unsqueeze(0)
            prob = model(crop)
            probability = round(sigmoid(float(prob[0][1])) , 1)
            prediction = prob.data.numpy().argmax()
#        print(y)
            if prediction == 1:
#            print(c2)
#            print(c1)
                L.append([cropped_image,x,y,probability])
                c2 += 1
            y += int(size_of_images*(1-overlapping_percentage))
        x += int(size_of_images*(1-overlapping_percentage))
    return(L)

def plot_probability_of_detection(image , color):

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 20))
    ax.imshow(image)
    c = 0
    for r in L:
        x , y = r[1] , r[2]
        c += 1
        rect = mpatches.Rectangle((y - int(size_of_images/2) , x - int(size_of_images/2)), size_of_images , size_of_images, fill=False, edgecolor = 'yellow' , linewidth=1)
        ax.add_patch(rect)
        plt.text(y , x , str(r[3]) , bbox={'facecolor': color})
    plt.show()


#def plot_probability_heatmap(image , L , size_of_images):
#    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 20))
#    ax.imshow(image)
#    c = 0
#    for r in L:
#        x , y = r[1] , r[2]
#        c += 1
#        rect = mpatches.Rectangle((y - int(size_of_images/2) , x - int(size_of_images/2)) , size_of_images , size_of_images , fill=True,  #alpha = 0.2, facecolor = ((r[3]- 0.5)*2 , 0 , 0) , linewidth=1)
#        ax.add_patch(rect)
#        plt.text(y , x , str(r[3]) , bbox={'facecolor': color})
#    plt.show()
    
def plot_probability_heatmap(image , L , size_of_images):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 20))
    ax.imshow(image)
    c = 0
    for r in L:
        x , y = r[1] , r[2]
        c += 1
        rect = mpatches.Rectangle((y - int(size_of_images/2) , x - int(size_of_images/2)) , size_of_images , size_of_images , fill=True,  alpha = r[3]/10, facecolor = (r[3] , 0 , 0) , linewidth=1)
        ax.add_patch(rect)
#        plt.text(y , x , str(r[3]) , bbox={'facecolor': color})
    plt.show()    

#We use the list L generated by slising_window, so the coordinate of the rectangles are L[i][1] and L[i][2]
    
    
#def detection_strategy(sliding_window , image , size_of_images , minimum_probability , color):
#    clusters = []
#    L = sliding_window
#    for crop in L:
#        p = crop[3]
#        if p > minimum_probability:
#            x , y = crop[1] , crop[2]
#            Poly = Polygon([(y - int(size_of_images/2) , x - int(size_of_images/2)) , (y - int(size_of_images/2) , x + int(size_of_images/2)) , (y + int(size_of_images/2) , x + int(size_of_images/2)) , (y + int(size_of_images/2) , x - int(size_of_images/2))])
#            for i in range(len(clusters)):
#                P = clusters[i]
#                if Poly.intersects(P):
#                    union = [Poly, P]
#                    new_P = cascaded_union(union)
#                    clusters[i] = new_P
#    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 20))
#    ax.imshow(image)
#    for cluster in clusters:
#        vertices = polygon.exterior.coords
#        poly = mpatches.Polygon(vertices , size_of_images, fill=False, edgecolor=color, linewidth=1)
#        ax.add_patch(poly)
#        plt.text(y , x , str(r[3]) , bbox={'facecolor': 'yellow' })

#    plt.show()
    
def detection_strategy(sliding_window , image , size_of_images , minimum_probability , color):
#    x , y = sliding_window[0][1] , sliding_window[0][2]
#    clusters = [Polygon([(y - int(size_of_images/2) , x - int(size_of_images/2)) , (y - int(size_of_images/2) , x + int(size_of_images/2)) , (y + int(size_of_images/2) , x + int(size_of_images/2)) , (y + int(size_of_images/2) , x - int(size_of_images/2))])]
    L = sliding_window
    clusters = []
    for crop in L:
        p = crop[3]
        if p > minimum_probability:
            x , y = crop[1] , crop[2]
            Poly = Polygon([(y - int(size_of_images/2) - 1 , x - int(size_of_images/2) - 1)  , (y - int(size_of_images/2) - 1 , x + int(size_of_images/2) + 1) , (y + int(size_of_images/2) + 1 , x + int(size_of_images/2) + 1) , (y + int(size_of_images/2) + 1 , x - int(size_of_images/2) - 1)])
            belongs_to_a_cluster = False
            Intersected_clusters = []
            temp_idx = []
            for i in range(len(clusters)):
                P = clusters[i]
                if Poly.intersects(P):
                    Intersected_clusters.append(clusters[i])
                    temp_idx.append(i)
#                    union = [Poly, P]
#                    new_P = cascaded_union(union)
#                    print(new_P)
#                    clusters[i] = new_P
                    belongs_to_a_cluster = True
            if belongs_to_a_cluster == False:
                clusters.append(Poly)
            else:
                union = [Poly] + Intersected_clusters
                new_P = cascaded_union(union)
                clusters[temp_idx[0]] = new_P
                for idx in temp_idx[1:]:
                    clusters.remove(clusters[idx])
#                print(clusters)
#    for cluster in clusters:
        
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 20))
    ax.imshow(image)
    for cluster in clusters:
#        print(cluster)
        vertices = cluster.exterior.coords
        poly = mpatches.Polygon(vertices , size_of_images, fill=False, color=color, linewidth=1)
        ax.add_patch(poly)
#        plt.text(y , x , str(r[3]) , bbox={'facecolor': 'yellow' })

    plt.show()
                    
                