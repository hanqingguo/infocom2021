import torch

data = torch.load('../trained_model/enhance_my_voice/chkpt_201000.pt')['model']
torch.save(data.type(torch.float32), 'chkpt_enhance.pt') 

