import torchvision.transforms as transforms
import Augmentor


#The Imagenet values for mean std are mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], recomputing them for multi_size_images folder give us values of [0.4050, 0.2958, 0.3421], std=[0.6287, 0.6657, 0.6585]. Those values should be recomputed when the dataset is modified, in order for the loss function the converge better during training.

#Apart from the transforms.ToTensor(), all the transformations can be modified


data_transforms = transforms.Compose([
#    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4050, 0.2958, 0.3421],
                                 std=[0.6287, 0.6657, 0.6585])
])


    
    
data_transforms_aug = transforms.Compose([
    transforms.Resize((300, 300)),
#    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    transforms.RandomRotation(0,30),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.RandomHorizontalFlip(p=0.15),
    transforms.ToTensor(),
#    transforms.ColorJitter(),
#    transforms.RandomAffine(30, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

data_transforms_color = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.1),
#    transforms.RandomRotation(0,30),
#    transforms.RandomVerticalFlip(p=0.4),
#    transforms.RandomHorizontalFlip(p=0.15),
    transforms.ToTensor(),
#    transforms.RandomAffine(30, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

#p = Augmentor.Pipeline()
#p.greyscale(probability=1)
#p.zoom(probability=1, min_factor=1.0, max_factor=1.0)
#p.rotate_random_90(probability=1)

data_transforms_global_resize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.1),
    transforms.RandomRotation(0,30),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.RandomHorizontalFlip(p=0.15),
    transforms.ToTensor(),
#    transforms.RandomAffine(30, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

data_transforms_global = transforms.Compose([
    transforms.Resize((400, 400)),
#    p.torch_transform()
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.1),
    transforms.RandomRotation(0,30),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.RandomHorizontalFlip(p=0.15),
    transforms.ToTensor(),
#    transforms.RandomAffine(30, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
    transforms.Normalize(mean=[0.4050, 0.2958, 0.3421],
                                 std=[0.6287, 0.6657, 0.6585])
])

data_transforms_big = transforms.Compose([
    transforms.Resize((480, 480)),
#    p.torch_transform()
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.1),
    transforms.RandomRotation(0,30),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.RandomHorizontalFlip(p=0.15),
    transforms.ToTensor(),
#    transforms.RandomAffine(30, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
#    transforms.Normalize(mean=[0.4238, 0.3234, 0.3791],
#                                 std=[0.229, 0.224, 0.225])
    transforms.Normalize(mean=[0.6702, 0.8248, 0.4805],
                                 std=[0.5260, 0.5598, 0.5536])
])

data_transforms_small = transforms.Compose([
    transforms.Resize((200, 200)),
#    p.torch_transform()
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.1),
    transforms.RandomRotation(0,30),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.RandomHorizontalFlip(p=0.15),
    transforms.ToTensor(),
    transforms.RandomAffine(30, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
    transforms.Normalize(mean=[0.6702, 0.8248, 0.4805],
                                 std=[0.5260, 0.5598, 0.5536])
])

#])