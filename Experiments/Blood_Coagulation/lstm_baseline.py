from NESDE import HyperNESDE, train_contextual_NESDE, eval_contextual_NESDE, plot_NESDE, Dataloader
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import os
import shutil

class BaseLSTM(nn.Module):
    def __init__(self,context_dim, state_dim,control_dim, device, cont_hidden=[32,32], lstm_hidden=64,dt=10000):
        super(BaseLSTM,self).__init__()
        context_sizes = [context_dim] + cont_hidden
        context_layers = []
        for i in range(1,len(context_sizes)):
            context_layers.extend([nn.Linear(context_sizes[i-1],context_sizes[i]),nn.ReLU()])
        context_layers.extend([nn.Linear(context_sizes[-1],16),nn.Tanh()])
        self.context_fc = nn.Sequential(*context_layers)
        self.lstm = nn.LSTM(input_size=16 + control_dim + state_dim + 1, hidden_size=lstm_hidden, num_layers=2, dropout=0.2)
        self.fc_out = nn.Sequential(*[nn.Linear(lstm_hidden,1),nn.ReLU()]) 
        self.context = None
        self.device = device
        self.lstm_state = None
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.dt = dt * torch.ones(1,device=device)
        self.n = 0
        self.m = state_dim
        self.loss_fn = nn.MSELoss()

    def forward(self,state,state_var,dt,U):
        control = U
        if state is None:
            self.lstm_state = None
            state = torch.zeros(self.state_dim,device=self.device)
        if self.context is None:
            context = torch.zeros(16,device=self.device)
        else:
            context = self.context
        n_steps = ((1.0/self.dt) * dt).type(torch.long)
        fract = ((1.0/self.dt) * dt) - n_steps.type(torch.float)
        for t in range(n_steps.item()):
            lstm_input = torch.cat([context.view(-1,16),control.view(-1,self.control_dim),state.view(-1,self.state_dim),self.dt.view(-1,1)],dim=1).unsqueeze(0)
            if self.lstm_state is None:
                lstm_out, self.lstm_state = self.lstm(lstm_input)
            else:
                lstm_out, self.lstm_state = self.lstm(lstm_input, self.lstm_state)

            state = 200*self.fc_out(lstm_out.squeeze(0))
        lstm_input = torch.cat([context.view(-1,16),control.view(-1,self.control_dim),state.view(-1,self.state_dim),fract.view(-1,1)],dim=1).unsqueeze(0)
        if self.lstm_state is None:
            lstm_out, self.lstm_state = self.lstm(lstm_input)
        else:
            lstm_out, self.lstm_state = self.lstm(lstm_input, self.lstm_state)

        state = 200*self.fc_out(lstm_out.squeeze(0))
        return state, None
    
    def set_context(self,context):
        if context is None:
            self.context = context
        else:
            self.context = self.context_fc(context.to(self.device))

    def reset_context(self):
        self.context = None
    
    def get_prior(self):
        return self.forward(None,None,0,torch.zeros(self.control_dim,device=self.device))


def load_data(data_path, group, seed=None):
    if seed is not None:
        with open(f"{data_path}_{group}_{seed:d}.pkl",'rb') as f:
            data = pickle.load(f)['data']
    else:
        with open(f"{data_path}_{group}.pkl", 'rb') as f:
            data = pickle.load(f)['data']
    return data

def main(train_size, valid_size, test_size=None, n_epochs=200, patience=50, device='cuda', n_seeds=1, sep_datasets=False,
         data_path="./Data/Blood_Coagulation_Data", outs_path="./Results", plot=False, plot_res=0.05, title=None):
    data = load_data(data_path,'train', seed=None)
    if title is None:
        title = f'n{train_size+valid_size:d}'
    device = torch.device(device)
    for seed in range(n_seeds):
        if n_seeds > 1:
            print("Seed #" + str(seed) + ":")
        # path for model weights:
        weights_path = outs_path + f"/{title}_model_w_" + str(seed) + ".pt"
        # path to save results:
        res_path = outs_path + f"/{title}_results_" + str(seed) + ".pkl"
        # path to save plots (directory):
        plots_path = outs_path + f"/{title}_graph_res_" + str(seed) + "/"


        # initiate dataloader, model, optimizer etc.
        dl = Dataloader(data, split=[train_size,valid_size,0], batch_size=50, shuffle=True,
                        global_shuffle=(n_seeds>1), verbose=True)
        model = BaseLSTM(42, 1, 1, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = None #torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98) # optional, set to None to disable

        # train the model:
        train_res = train_contextual_NESDE(model, dl, optim=optimizer, train_epochs=n_epochs, w_path=weights_path,
                                           random_samples=2, eval_rate=1, patience=patience, sched=scheduler, verbose=1, deterministic=True)

        # evaluate on test set:
        data = load_data(data_path, 'test', seed=None)
        if test_size is None:
            test_size = len(data)
        dl = Dataloader(data, split=[0,0,test_size], batch_size=10, shuffle=False, global_shuffle=False, verbose=False)
        dl.test()
        test_nll, test_mse, test_corr = eval_contextual_NESDE(model, dl, deterministic=True)
        print("Test loss: NLL: " + str(np.mean(test_nll)) + ", MSE: " + str(np.mean(test_mse)))
        # save results:
        with open(res_path,'wb') as f:
            pickle.dump({'train_summary':train_res,'NLL':test_nll,'MSE':test_mse,'CORR':test_corr},f)

        if plot:
            try:
                os.mkdir(outs_path + f"/{title}_graph_res_" + str(seed))
            except:
                shutil.rmtree(outs_path + f"/{title}_graph_res_" + str(seed))
                os.mkdir(outs_path + f"/{title}_graph_res_" + str(seed))
            # plot results and save them:
            plot_NESDE(model,dl,dt=plot_res,path=plots_path,plot_control=True,pw_control=True,contextual=True)

        return test_nll, test_mse, test_corr

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
    for i in range(n_exps):
        kwargs_i = {k:v[i] for k,v in kwargs.items()}
        nlls, mses, corrs = main(**kwargs_i)
        res = pd.concat((res, pd.DataFrame(dict(
            train_size = kwargs_i['train_size']+kwargs_i['valid_size'],
            NLL = nlls,
            MSE = mses,
            CORR = corrs
        ))))

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help="pickled data path", default="./Data/Blood_Coagulation")
    parser.add_argument('--device', help="cuda/cpu", default="cuda")
    parser.add_argument('--outs_path', type=str, help="Where to save outputs", default="./Results_LSTM/")
    parser.add_argument('--plot', help="to plot the whole test set",default = True)
    parser.add_argument('--plot_res', help="time resolution for plotting",default = 1.0)
    parser.add_argument('--n_seeds', help="number of seeds to run", default = 1)
    parser.add_argument('--sep_datasets', help="use separate datasets for the seeds, requires n_seeds datasets", default=False)
    parser.add_argument('--train_size', help="train dataset size", default = 5.0/6.0)
    parser.add_argument('--valid_size', help="validation dataset size", default = 1.0/6.0)
    parser.add_argument('--test_size', help="test dataset size", default = None)
    parser.add_argument('--patience', help="stopping criteria", default = 40)
    parser.add_argument('--n_epochs', help="maximal number of epochs to train", default = 200)
    args = parser.parse_args()

    try:
        os.mkdir(args.outs_path)
    except:
        print("Directory exists, continues...")
    main(args.train_size, args.valid_size, args.test_size, args.n_epochs, args.patience, args.device, args.n_seeds,
         args.sep_datasets, args.data_path, args.outs_path, args.plot, args.plot_res)
