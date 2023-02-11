from NESDE import  Dataloader, train_NESDE, eval_NESDE
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class BaseLSTM(nn.Module):
    def __init__(self,state_dim,control_dim, device, lstm_hidden=32):
        super(BaseLSTM,self).__init__()
        self.lstm = nn.LSTM(input_size= control_dim + state_dim + 1, hidden_size=lstm_hidden, num_layers=2, dropout=0.2)
        self.fc_out = nn.Sequential(*[nn.Linear(lstm_hidden,lstm_hidden),nn.Tanh(),nn.Linear(lstm_hidden,1)])
        self.device = device
        self.lstm_state = None
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.n = 0
        self.dt = True
        self.m = state_dim
        self.loss_fn = nn.MSELoss()
        self.B = None
        self.dtr = torch.Tensor([0.01]).to(device)

    def forward(self,state,state_var,dt,U):
        control = U
        if state is None:
            self.lstm_state = None
            state = torch.zeros(self.state_dim,device=self.device)
        lstm_input = torch.cat([control.view(-1,self.control_dim).to(self.device),state.view(-1,self.state_dim).to(self.device),dt.view(-1,1).to(self.device)],dim=1).unsqueeze(0)
        if self.lstm_state is None:
            lstm_out, self.lstm_state = self.lstm(lstm_input)
        else:
            lstm_out, self.lstm_state = self.lstm(lstm_input, self.lstm_state)

        state = self.fc_out(lstm_out.squeeze(0))
        return state, None

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

def load_data(data_path, group, seed=None):
    pth = f"{data_path}"
    if seed is not None: pth += f'_{seed:d}'
    pth += f'_{group}.pkl'
    with open(pth, 'rb') as f:
        data = pickle.load(f)['data']

    return data

def main(train_size, valid_size, test_size, n_epochs, patience, lr, do_train,
         data_path, outs_path, sep_datasets=False, title=None, device='cuda', exp_id=0,
         batch_size=50, random_samples=5, train_by_nll=True, valid_by_nll=True, n_samp_per_traj=None, Tmin=None, gamma=0.995,
         lstm=False, ood_test=False, log_rate=10, skip_first=False):
    data_size = train_size + valid_size
    if title is None:
        title = f'n{train_size + valid_size:d}'
    device = torch.device(device)
    with open(data_path + "_meta_train.pkl", 'rb') as f:
        meta = pickle.load(f)
    weights_path = outs_path + f"{title}_model_w_" + (f"lstm_" if lstm else "") + str(exp_id) + ".pt"
    # path to save results:
    res_path = outs_path + f"{title}_results_" + (f"lstm_" if lstm else "") + str(exp_id) + ".pkl"

    # load data:
    data = load_data(data_path, 'train', exp_id if sep_datasets else None)
    assert data_size <= len(data)

    # initiate dataloader, model, optimizer etc.
    dl = Dataloader(data, split=[train_size, valid_size, 0], batch_size=batch_size, shuffle=True,
                    global_shuffle=True, verbose=True)
    model = BaseLSTM(1, 1, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma) \
        if (gamma is not None and 0 < gamma < 1) else None

    # train the model:
    train_res = None
    if do_train:
        train_res = train_NESDE(model, dl, optim=optimizer, train_epochs=n_epochs, w_path=weights_path,
                                    random_samples=random_samples, eval_rate=1, patience=patience,
                                    train_nll=train_by_nll, valid_by_nll=valid_by_nll,
                                    sched=scheduler, Tmin=Tmin, n_samp_per_traj=n_samp_per_traj, verbose=1,
                                    log_rate=log_rate, skip_first=skip_first, deterministic=True, irreg_lstm=True)
    else:
        model.load_state_dict(torch.load(weights_path, map_location=device))

    # evaluate on test set:
    data = load_data(data_path, 'test', exp_id if sep_datasets else None)
    dl = Dataloader(data, split=[0, 0, test_size], batch_size=batch_size, shuffle=False, global_shuffle=False,
                    verbose=False)
    dl.test()
    test_nll, test_mse, test_corr, ttimes, tnlls, tmses, lambdas = eval_NESDE(
            model, dl, Tmin=Tmin, n_samp_per_traj=n_samp_per_traj, verbose=1, detailed_res=True, skip_first=skip_first, deterministic=True, irreg_lstm=True)
    # evaluate on ood set:
    if ood_test:
        data = load_data(data_path, 'ood', exp_id if sep_datasets else None)
        dl = Dataloader(data, split=[0, 0, test_size], batch_size=batch_size, shuffle=False, global_shuffle=False,
                        verbose=False)
        dl.test()
        ood_nll, ood_mse, ood_corr, ood_ttimes, ood_tnlls, ood_tmses, ood_lambdas = eval_NESDE(
                model, dl, Tmin=Tmin, n_samp_per_traj=n_samp_per_traj, verbose=1, detailed_res=True, skip_first=skip_first, deterministic=True, irreg_lstm=True)
    # save results:
    if do_train:
        with open(res_path, 'wb') as f:
            if ood_test:
                pickle.dump({'train_summary': train_res, ('VAR' if lstm else 'NLL'): test_nll,
                             'MSE': test_mse, 'CORR': test_corr, 'lambdas': lambdas,
                             ('OOD_VAR' if lstm else 'OOD_NLL'): test_nll, 'OOD_MSE': test_mse, 'OOD_CORR': test_corr,
                             'ood_lambdas': lambdas}, f)
            else:
                pickle.dump({'train_summary': train_res, ('VAR' if lstm else 'NLL'): test_nll,
                             'MSE': test_mse, 'CORR': test_corr, 'lambdas': lambdas}, f)

    if ood_test:
        return test_nll, test_mse, test_corr, ttimes, tnlls, tmses, ood_nll, ood_mse, ood_corr, ood_ttimes, ood_tnlls, ood_tmses
    else:
        return test_nll, test_mse, test_corr, ttimes, tnlls, tmses


def run_multi_experiments(n_exps, **kwargs):
    # args for main() can be given as a tuple/list (one value per experiment) or anything else (constant over exps)

    # preprocess args
    ood_testl = [False] * n_exps
    for k, v in kwargs.items():
        if not isinstance(v, (list, tuple)):
            # convert to list
            kwargs[k] = n_exps * [v]
        else:
            # assert list len
            if len(v) != n_exps:
                raise ValueError(f'Argument {k} with {len(v)} values does not fit {n_exps} experiments.')
        if k == 'ood_test':
            ood_testl = kwargs[k]
            assert (np.array(ood_testl).sum() == 0) or (np.array(ood_testl).sum() == n_exps)

    # run experiments
    res_per_t = pd.DataFrame()
    res_per_t_ood = pd.DataFrame()
    for i in range(n_exps):
        print("Starting experiment #" + str(i) + '...')
        kwargs_i = {k: v[i] for k, v in kwargs.items()}
        kwargs_i['exp_id'] = i
        if ood_testl[i]:
            nlls, mses, corrs, ttimes, tnlls, tmses, ood_nlls, ood_mses, ood_corrs, ood_ttimes, ood_tnlls, ood_tmses = main(**kwargs_i)
            c_res_per_t_ood = pd.DataFrame(dict(
                train_size=kwargs_i['train_size'] + kwargs_i['valid_size'],
                ood_time=ood_ttimes,
                OOD_NLL=ood_tnlls,
                OOD_MSE=ood_tmses,
            ))
            res_per_t_ood = pd.concat((res_per_t_ood, c_res_per_t_ood))
            c_res_per_t = pd.DataFrame(dict(
                train_size=kwargs_i['train_size'] + kwargs_i['valid_size'],
                time=ttimes,
                NLL=tnlls,
                MSE=tmses,
            ))
            res_per_t = pd.concat((res_per_t, c_res_per_t))
        else:
            nlls, mses, corrs, ttimes, tnlls, tmses = main(**kwargs_i)
            c_res_per_t = pd.DataFrame(dict(
                train_size=kwargs_i['train_size'] + kwargs_i['valid_size'],
                time=ttimes,
                NLL=tnlls,
                MSE=tmses,
            ))
            res_per_t = pd.concat((res_per_t, c_res_per_t))
        c_res_per_t.to_csv(kwargs_i['outs_path'] + 'table_res_per_t_exp' + str(i) + '.csv', index=False)
        if kwargs['ood_test'][0]:
            c_res_per_t_ood.to_csv(kwargs_i['outs_path'] + 'table_res_per_t_ood_exp' + str(i) + '.csv', index=False)
    return res_per_t_ood, res_per_t



