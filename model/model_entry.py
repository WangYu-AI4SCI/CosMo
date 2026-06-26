
import torch.nn as nn
import torch
from model.sota.fcn import MultiViewNet


def select_model(args):
	type2model = {
		'MultiViewNet':
            MultiViewNet(
			embed_dim=args.embed_dim if hasattr(args, 'embed_dim') else 256),
	}
	model = type2model[args.model_type]
	return model


def equip_multi_gpu(model, args):
	model = nn.DataParallel(model, device_ids=args.gpus)
	return model