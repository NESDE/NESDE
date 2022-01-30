from NESDE import NESDE, train_NESDE, eval_NESDE, train_LSTM, eval_LSTM, plot_NESDE, Dataloader, ControlCall
import pickle, time
import numpy as np
import pandas as pd
import torch
import argparse
import os
import shutil
import torch
from torch import nn

def find_nearest(a, values):
    ids = [(np.abs(v-a)).argmin() for v in values]
    return ids

def load_whole_data(data_path, group, seed=None):
    pth = f"{data_path}"
    if seed is not None: pth += f'_{seed:d}'
    pth_whole = pth + f'_whole_{group}.pkl'
    with open(pth_whole, 'rb') as f:
        data_whole = pickle.load(f)['data']
    return data_whole

def load_data(data_path, group, meta=None, device='cuda', seed=None, const_steps_subsample=None, verbose=1):
    if meta is None:
        with open(f"{data_path}_meta_{group}.pkl", 'rb') as f:
            meta = pickle.load(f)
    pth = f"{data_path}"
    if seed is not None: pth += f'_{seed:d}'
    pth_whole = pth + f'_whole_{group}.pkl'
    pth += f'_{group}.pkl'
    with open(pth, 'rb') as f:
        data = pickle.load(f)['data']

    if const_steps_subsample is not None:
        with open(pth_whole, 'rb') as f:
            data_whole = pickle.load(f)['data']
        data_whole = [{k:v[::const_steps_subsample] for k,v in d.items()} for d in data_whole]
        # set observable points
        for d,d_obs in zip(data_whole,data):
            d['obs'] = d['obs'][:, 0].view(-1, 1)
            d['mask'] = torch.zeros((len(d['times']),1))
            observable_ids = find_nearest(d['times'], d_obs['times'])
            d['mask'][[observable_ids]] = 1
        data = data_whole
        if verbose >= 1:
            print(f'Loaded data with subsampling 1:{const_steps_subsample:d}. '
                  f'{data[0]["mask"].sum():.0f}/{len(data[0]["mask"]):d}={100*data[0]["mask"].mean():.1f}% '
                  f'of the points are observable.')
            print(len(data), {k:v.shape for k,v in data[5].items()})

    # create control as callable:
    res_fac = 1 if const_steps_subsample is None else const_steps_subsample
    for i in range(len(data)):
        data[i]['U'] = ControlCall(data[i]['U'].to(device), lambda U, t: U[
            (t * ((meta['res'] - 1) / res_fac / (meta['tn'] - meta['t0']))).type(torch.long)])

    return data

def main(train_size, valid_size, test_size, n=2, n_epochs=200, patience=50, lr=1e-2, device='cuda', n_seeds=1, sep_datasets=False, do_train=True,
         data_path="./Data/SDE_control_complex", outs_path="./Data_efficiency/results_NESDE", plot=False, plot_res=0.05, title=None,
         batch_size=10, random_samples=5, train_by_nll=True, valid_by_nll=True, n_samp_per_traj=None, Tmin=None, gamma=0.995, stationary_model=False,
         const_steps_res=None, lstm=False):
    data_size = train_size + valid_size
    if title is None:
        title = f'n{train_size+valid_size:d}'
    device = torch.device(device)
    with open(data_path + "_meta_train.pkl", 'rb') as f:
        meta = pickle.load(f)
    dt = (meta['tn'] - meta['t0']) / (meta['res'] - 1)
    for seed in range(n_seeds):
        if n_seeds > 1:
            print("Seed #" + str(seed) + ":")
        # path for model weights:
        weights_path = outs_path + f"{title}_model_w_" + (f"lstm_" if lstm else "") + str(seed) + ".pt"
        # path to save results:
        res_path = outs_path + f"{title}_results_" + (f"lstm_" if lstm else "") + str(seed) + ".pkl"
        # path to save plots (directory):
        plots_path = outs_path + f"{title}_graph_res_" + (f"lstm_" if lstm else "") + str(seed) + "/"

        # load data:
        data = load_data(data_path, 'train', meta, device,
                         seed if sep_datasets else None, const_steps_res)
        assert data_size <= len(data)

        # initiate dataloader, model, optimizer etc.
        dl = Dataloader(data, split=[train_size,valid_size,0], batch_size=batch_size, shuffle=True,
                        global_shuffle=(n_seeds>1), verbose=True)
        if lstm:
            model = LSTM(device=device)
        else:
            model = NESDE(n,1,params_interval=100,device=device,control_dim=1,lambda_hidden=[16,16],V_hidden=[16,16],Q_hidden=[16,16],
                          complex=True,dt=dt,nonlinearity=torch.nn.ReLU,stable=True,B=None, lambdas_fac=None, dropout=0.1,
                          stationary=stationary_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma) \
            if (gamma is not None and 0<gamma<1) else None

        # train the model:
        train_res = None
        if do_train:
            if lstm:
                train_res = train_LSTM(model, dl, optim=optimizer, train_epochs=n_epochs, w_path=weights_path,
                                       eval_rate=1, patience=patience, sched=scheduler, verbose=1)
            else:
                train_res = train_NESDE(model, dl, optim=optimizer, train_epochs=n_epochs, w_path=weights_path,
                                        random_samples=random_samples, eval_rate=1, patience=patience,
                                        train_nll=train_by_nll, valid_by_nll=valid_by_nll,
                                        sched=scheduler, Tmin=Tmin, n_samp_per_traj=n_samp_per_traj, verbose=1)
        else:
            model.load_state_dict(torch.load(weights_path))

        # evaluate on test set:
        data = load_data(data_path, 'test', meta, device,
                         seed if sep_datasets else None, const_steps_res)
        dl = Dataloader(data, split=[0,0,test_size], batch_size=batch_size, shuffle=False, global_shuffle=False, verbose=False)
        dl.test()
        if lstm:
            test_nll, test_mse, test_corr, ttimes, tnlls, tmses = eval_LSTM(model, dl, Tmin=Tmin, verbose=1, detailed_res=True)
            lambdas = None
        else:
            test_nll, test_mse, test_corr, ttimes, tnlls, tmses, lambdas = eval_NESDE(
                model, dl, Tmin=Tmin, n_samp_per_traj=n_samp_per_traj, verbose=1, detailed_res=True, skip_first=True)
        # save results:
        if do_train:
            with open(res_path,'wb') as f:
                pickle.dump({'train_summary':train_res,('VAR' if lstm else 'NLL'):test_nll,
                             'MSE':test_mse,'CORR':test_corr,'lambdas':lambdas},f)

        if plot:
            try:
                os.mkdir(outs_path + f"{title}_graph_res_" + str(seed))
            except:
                shutil.rmtree(outs_path + f"{title}_graph_res_" + str(seed))
                os.mkdir(outs_path + f"{title}_graph_res_" + str(seed))
            # plot results and save them:
            plot_NESDE(model,dl,dt=plot_res,path=plots_path,
                       whole_data=load_whole_data(data_path, 'test', seed if sep_datasets else None))

        return test_nll, test_mse, test_corr, ttimes, tnlls, tmses

def run_multi_experiments(n_exps, **kwargs):
    # args for main() can be given as a tuple/list (one value per experiment) or anything else (constant over exps)

    # preprocess args
    for k, v in kwargs.items():
        if not isinstance(v, (list, tuple)):
            # convert to list
            kwargs[k] = n_exps * [v]
        else:
            # assert list len
            if len(v) != n_exps:
                raise ValueError(f'Argument {k} with {len(v)} values does not fit {n_exps} experiments.')

    # run experiments
    res = pd.DataFrame()
    res_per_t = pd.DataFrame()
    for i in range(n_exps):
        print()
        kwargs_i = {k:v[i] for k,v in kwargs.items()}
        try:
            os.mkdir(kwargs_i['outs_path'])
        except:
            a=None
        nlls, mses, corrs, ttimes, tnlls, tmses = main(**kwargs_i)
        res = pd.concat((res, pd.DataFrame(dict(
            train_size = kwargs_i['train_size']+kwargs_i['valid_size'],
            NLL = nlls,
            MSE = mses,
            CORR = corrs
        ))))
        res_per_t = pd.concat((res_per_t, pd.DataFrame(dict(
            train_size = kwargs_i['train_size']+kwargs_i['valid_size'],
            time = ttimes,
            NLL = tnlls,
            MSE = tmses,
        ))))
    return res, res_per_t


class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, output_size=1, device='cuda'):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, self.hidden_size, device=self.device)
        self.linear = nn.Linear(self.hidden_size, output_size, device=self.device)

        self.hidden_cell = None

        self.init()

    def init(self):
        self.hidden_cell = (torch.zeros(1,1,self.hidden_size).to(self.device),
                            torch.zeros(1,1,self.hidden_size).to(self.device))

    def get_LSTM_input(self, traj, Tmax=None, remove_last=True):
        u = traj['U'](traj['times']).to(self.device)
        z = (traj['mask'] * traj['obs']).to(self.device)
        is_observed = traj['mask'].to(self.device)
        if Tmax is not None:
            times = traj['times'].to(self.device)
            is_observed = is_observed * (times<=Tmax)
            z = z * is_observed
        if remove_last:
            u, z, is_observed = u[:-1], z[:-1], is_observed[:-1]
        x = torch.cat((u, z, is_observed), dim=1)
        return x

    def forward(self, input_seq, Tmax=None):
        if isinstance(input_seq, dict):
            input_seq = self.get_LSTM_input(input_seq, Tmax)
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions  # (seq_len, output_dim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help="pickled data path", default="./Data/SDE_control_complex")
    parser.add_argument('--device', help="cuda/cpu", default="cuda")
    parser.add_argument('--outs_path', type=str, help="Where to save outputs", default="./Data_efficiency/results_NESDE/")
    parser.add_argument('--plot', help="to plot the whole test set",default = False)
    parser.add_argument('--plot_res', help="time resolution for plotting",default = 0.05)
    parser.add_argument('--n_seeds', help="number of seeds to run", default = 1)
    parser.add_argument('--sep_datasets', help="use separate datasets for the seeds, requires n_seeds datasets", default=False)
    parser.add_argument('--train_size', help="train dataset size", default = 800)
    parser.add_argument('--valid_size', help="validation dataset size", default = 200)
    parser.add_argument('--test_size', help="test dataset size", default = 2000)
    parser.add_argument('--patience', help="stopping criteria", default = 20)
    parser.add_argument('--n_epochs', help="maximal number of epochs to train", default = 200)
    args = parser.parse_args()

    try:
        os.mkdir(args.outs_path)
    except:
        print("Directory exists, continues...")
    main(args.train_size, args.valid_size, args.test_size, args.n_epochs, args.patience, args.device, args.n_seeds,
         args.sep_datasets, args.data_path, args.outs_path, args.plot, args.plot_res)
