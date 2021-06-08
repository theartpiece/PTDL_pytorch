import torch

class ProtoTypeDL(torch.nn.Module):
    def __init__(self,args,k,m=15):
        super().__init__()
        self.ae_layer=CNNAutoencoder(args)
        self.p=self.ae_layer.enc_output_size
        self.proto_classifier=ProtoClassifier(args,m,k,self.p)

    def loss(self,batch,y,l=0.05,l1=0.05,l2=0.05):
        enc_output=self.ae_layer(batch)
        self.ae_loss=self.ae_layer.loss(l)
        self.class_loss=self.proto_classifier.loss(enc_output,y,l1,l2)

        return self.class_loss+self.ae_loss

class ProtoClassifier(torch.nn.Module):
    def __init__(self,args,m,k,p):
        super().__init__()
        self.skip_proto_layer=args.skip_proto_layer
        if args.skip_proto_layer:
            self.proto_space_project=torch.nn.Linear(p,m)
        else:
            self.prototypes=torch.nn.Parameter(torch.rand(m,p).cuda(),requires_grad=True)
        self.weights_project=torch.nn.Linear(m,k,bias=False)
        self.p=p
        self.class_loss_fn = torch.nn.CrossEntropyLoss()

    def loss(self,enc_output,y,l1,l2):
        total_loss=0
        if self.skip_proto_layer:
            encd_proto_distance=self.proto_space_project(enc_output.view(-1,self.p))
        else:
            encd_proto_distance=torch.cdist(enc_output.view(-1,self.p),self.prototypes)

        self.proto_classifier_output=self.weights_project(encd_proto_distance)
        self.class_loss = self.class_loss_fn(self.proto_classifier_output, y)
        total_loss+=self.class_loss
        if not self.skip_proto_layer:
            self.r1_loss = torch.mean(torch.min(encd_proto_distance, dim=0)[0])
            self.r2_loss = torch.mean(torch.min(encd_proto_distance, dim=1)[0])
            total_loss+=l1*self.r1_loss
            total_loss+=l2*self.r2_loss
        return total_loss


class Print(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input):
        print(input.size())
        return input
class CNNAutoencoder(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.skip_decode=args.skip_decode
        self.activation_fn=torch.nn.ReLU if self.skip_decode else torch.nn.Sigmoid
        self.enc_layers=torch.nn.Sequential(
            torch.nn.Conv2d(1,32,(3,3),stride=2,padding=1),
            # Print(),
            self.activation_fn(),
            torch.nn.Conv2d(32, 32, (3, 3), stride=2,padding=1),
            # Print(),
            self.activation_fn(),
            torch.nn.Conv2d(32, 32, (3, 3), stride=2,padding=1),
            # Print(),
            self.activation_fn(),
            torch.nn.Conv2d(32, 10, (3, 3), stride=2,padding=1),
            # Print(),
            self.activation_fn()
        )
        self.enc_output_size=40
        if not self.skip_decode:
            self.loss_fn= torch.nn.MSELoss(reduction='sum')

            self.dec_layers=torch.nn.Sequential(
                torch.nn.ConvTranspose2d(10,32,(3,3),stride=2,padding=1,output_padding=1),
                # Print(),
                torch.nn.Sigmoid(),
                torch.nn.ConvTranspose2d(32, 32, (3, 3), stride=2,padding=1,output_padding=0),
                # Print(),
                torch.nn.Sigmoid(),
                torch.nn.ConvTranspose2d(32, 32, (3, 3), stride=2,padding=1,output_padding=1),
                # Print(),
                torch.nn.Sigmoid(),
                torch.nn.ConvTranspose2d(32, 1, (3, 3), stride=2,padding=1,output_padding=1),
                # Print(),
                torch.nn.Sigmoid()
            )

    def forward(self,batch,operation="encode"):
        # if operation=="encode":
        #     return self.enc_layers(batch)
        #
        # elif operation == "decode":
        #     return self.dec_layers(batch)
        self.batch_size=len(batch)
        self.batch=batch
        self.enc_output=self.encode(batch)
        self.dec_output=self.decode(self.enc_output)
        return self.enc_output
    def loss(self,l):
        if self.skip_decode:
            return 0
        else:
            return l*1. / self.batch_size * self.loss_fn(self.batch, self.dec_output)


    def encode(self,batch):
        # batch is (B, 28,28)
        return self.enc_layers(batch)
    def decode(self,batch):
        # batch is (B, 28,28)+l1*self.r1_loss+l2*self.r2_loss
        return self.dec_layers(batch)




