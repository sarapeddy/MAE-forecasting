import os
import torch


def save_model(args, model, optimizer):
    # out = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.current_epoch))
    # out = os.path.join(f'./{args.model_path}/forecasting', "Pretrained_{}_{}.pkl".format(
    #         args.dataset, args.emb_dim))

    if not os.path.exists(f'./{args.model_path}/forecasting/Pretrained_{args.dataset}_{args.emb_dim}'):
        os.makedirs(f'./{args.model_path}/forecasting/Pretrained_{args.dataset}_{args.emb_dim}', exist_ok=True)

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), f'./{args.model_path}/forecasting/Pretrained_{args.dataset}_{args.emb_dim}.pkl')
    else:
        torch.save(model.state_dict(), f'./{args.model_path}/forecasting/Pretrained_{args.dataset}_{args.emb_dim}.pkl')
