import os
import torch


def save_model(args, model, mode, pred_len, optimizer):
    if not os.path.exists(f'./{args.model_path}/forecasting/B{args.batch_size}_E{args.emb_dim}/{mode}/Pretrained_{args.dataset}_{args.emb_dim}_{pred_len}'):
        os.makedirs(f'./{args.model_path}/forecasting/B{args.batch_size}_E{args.emb_dim}/{mode}/Pretrained_{args.dataset}_{args.emb_dim}_{pred_len}', exist_ok=True)

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), f'./{args.model_path}/forecasting/B{args.batch_size}_E{args.emb_dim}/{mode}/Pretrained_{args.dataset}_{args.emb_dim}_{pred_len}/Pretrained_{args.dataset}_{args.emb_dim}_{pred_len}.pkl')
    else:
        torch.save(model.state_dict(), f'./{args.model_path}/forecasting/B{args.batch_size}_E{args.emb_dim}/{mode}/Pretrained_{args.dataset}_{args.emb_dim}_{pred_len}/Pretrained_{args.dataset}_{args.emb_dim}_{pred_len}.pkl')
