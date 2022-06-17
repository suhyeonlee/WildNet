"""
Network Initializations
"""

import logging
import importlib
import torch
import datasets



def get_net(args, criterion, criterion_aux=None, cont_proj_head=0, wild_cont_dict_size=0):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(args=args, num_classes=datasets.num_classes,
                    criterion=criterion, criterion_aux=criterion_aux, cont_proj_head=cont_proj_head, wild_cont_dict_size=wild_cont_dict_size)
    num_params = sum([param.nelement() for param in net.parameters()])
    logging.info('Model params = {:2.3f}M'.format(num_params / 1000000))

    net = net.cuda()
    return net


def warp_network_in_dataparallel(net, gpuid):
    """
    Wrap the network in Dataparallel
    """
    # torch.cuda.set_device(gpuid)
    # net.cuda(gpuid)    
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpuid], find_unused_parameters=True)
    # net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpuid])#, find_unused_parameters=True)
    return net


def get_model(args, num_classes, criterion, criterion_aux=None, cont_proj_head=0, wild_cont_dict_size=0):
    """
    Fetch Network Function Pointer
    """
    network = args.arch
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(args=args, num_classes=num_classes, criterion=criterion, criterion_aux=criterion_aux, cont_proj_head=cont_proj_head, wild_cont_dict_size=wild_cont_dict_size)
    return net
