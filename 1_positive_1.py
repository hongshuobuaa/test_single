model_resnet = resnet50() #this is built completely same as  the link above
resnet50_weights = torch.load("resnet50-19c8e357.pth") 

#get all layer name from model_resnet
names = {}
for name,param in model_resnet.named_parameters():
    names[name] = 0

#to see if there's anything missing
for key in resnet50_weights:
    if key not in names:
        print(key)
