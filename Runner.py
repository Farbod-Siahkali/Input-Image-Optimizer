import torch
from input_optimizer import RegularizedClassSpecificImageGeneration
from torchvision import models

target_class = 150  #YOUR TARGET
pretrained_model = models.alexnet(pretrained=True)

from torchreid import models 

model = models.build_model(
    name='osnet_x1_0',
    num_classes=751,
    loss='softmax',
    pretrained=False
)

from models import CA_market_model2

attr_net_camarket = CA_market_model2(model=model,
                feature_dim = 512,
                num_id = 751,
                attr_dim = 46,
                need_id = False,
                need_attr = True,
                need_collection = False)

model_path = './result/best_attr_net.pth'
trained_net = torch.load(model_path)
attr_net_camarket.load_state_dict(trained_net.state_dict())

attr_net_camarket = attr_net_camarket.to('cuda')

csig = RegularizedClassSpecificImageGeneration(pretrained_model, target_class)
csig.generate()
