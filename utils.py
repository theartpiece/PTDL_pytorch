import torch
import numpy as np
from torch.nn.functional import grid_sample
class ElasticDeformation():
    @classmethod
    def initialize(cls,sigma=4,alpha=20):
        dim=6*sigma+1 if sigma>=1 else 3
        w_x=torch.exp(-torch.square(torch.arange(dim).view(1,-1)-dim/2)/(2*sigma*sigma))
        # w=(1/(2*np.pi*sigma*sigma))*w_x.t().mm(w_x)
#         cls.gauss=torch.nn.Conv2d(1,1,(dim,dim),padding=(dim//2,dim//2),bias=False)
#         cls.gauss.weight=torch.nn.Parameter(w.view(1,1,dim,dim),requires_grad=False)
        cls.gauss_y=torch.nn.Conv2d(1,1,(dim,1),padding=(dim//2,0),bias=False)
        cls.gauss_y.weight=torch.nn.Parameter(w_x.view(1,1,dim,1).cuda(),requires_grad=False)
        cls.gauss_x=torch.nn.Conv2d(1,1,(1,dim),padding=(0,dim//2),bias=False)
        cls.gauss_x.weight=torch.nn.Parameter(w_x.view(1,1,1,dim).cuda(),requires_grad=False)
        cls.alpha=alpha

    @classmethod
    def getElasticDeformation(cls,images):
        """
        Credits to https://github.com/developer0hye/Elastic-Distortion/blob/master/main.cpp
        :param images:
        :return:
        """
        B,C,H,W=images.shape
        x=torch.tile((2*torch.arange(W)/W-1).view(1,1,1,-1),(B,1,H,1)).cuda()
        y=torch.transpose(x,-1,-2).cuda()
        dx=cls.gauss_x((2*torch.rand(B,1,H,W)-1).cuda())
        dy=cls.gauss_y((2*torch.rand(B,1,H,W)-1).cuda())
        dx=cls.alpha*dx/torch.norm(dx,p=1)
        dy=cls.alpha*dy/torch.norm(dy,p=1)
        dxy=torch.cat(((x+dx).permute(0,2,3,1),(y+dy).permute(0,2,3,1)),dim=-1)
#         print(dxy[0])
        return torch.nn.functional.grid_sample(images,dxy)

def evaluate(eval_dler,model):
    model.eval()
    with torch.no_grad():
        total_loss=0
        sum_accuracy=0
        num_samples=0
        for test_batch in eval_dler:
            batch, y = test_batch[0].cuda(), test_batch[1].cuda()
            num_samples+=len(batch)
            loss=model.loss(batch,y)
            total_loss+=len(batch)*loss
            predict=torch.argmax(model.proto_classifier.proto_classifier_output, dim=-1)
            # actual=batch[1]
            sum_accuracy+=torch.sum(predict==y)
    return total_loss/num_samples,sum_accuracy/num_samples

