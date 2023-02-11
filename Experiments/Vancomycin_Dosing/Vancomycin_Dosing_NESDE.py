from NESDE import HyperNESDE, train_contextual_NESDE, eval_contextual_NESDE, plot_NESDE, Dataloader
import pickle
import numpy as np
import pandas as pd
import torch
import argparse
import os
import shutil



def load_data(data_path, group, seed=None):
    if seed is not None:
        with open(f"{data_path}_{group}_{seed:d}.pkl",'rb') as f:
            data = pickle.load(f)['data']
    else:
        with open(f"{data_path}_{group}.pkl", 'rb') as f:
            data = pickle.load(f)['data']
    return data

def main(train_size, valid_size, test_size=None, n_epochs=200, patience=50, device='cuda', n_seeds=1,
         data_path="./Data/Vancomycin_Dosing", outs_path="./Results", plot=False, plot_res=0.05, title=None, skip_first=True, do_train=True):
    for seed in range(n_seeds):
        data = load_data(data_path,'train', seed=None)
        if train_size < 1:
            train_size = int(len(data) * train_size)
            valid_size = int(len(data) * valid_size)
        if title is None:
            title = f'n{train_size+valid_size:d}'
        device = torch.device(device)
        if n_seeds > 1:
            print("Seed #" + str(seed) + ":")
        # path for model weights:
        weights_path = outs_path + f"/{title}_model_w_" + str(seed) + ".pt"
        # path to save results:
        res_path = outs_path + f"/{title}_results_" + str(seed) + ".pkl"
        # path to save plots (directory):
        plots_path = outs_path + f"/{title}_graph_res_" + str(seed) + "/"

        # initiate dataloader, model, optimizer etc.
        for di in range(len(data)):
            # add zero before any dosage:
            new_dat1 = torch.zeros((len(data[di]['times'])+1,1))
            new_dat1[1:] = data[di]['times']
            new_dat2 = torch.zeros((len(data[di]['obs'])+1,1))
            new_dat2[1:] = data[di]['obs']
            new_dat3 = torch.ones((len(data[di]['mask'])+1,1))
            new_dat3[1:] = data[di]['mask']
            data[di]['times'] = new_dat1
            data[di]['obs'] = new_dat2
            data[di]['mask'] = new_dat3


        dl = Dataloader(data, split=[train_size,valid_size,0], batch_size=50, shuffle=True,
                        global_shuffle=(n_seeds>1), verbose=True)
        if do_train:
            model_kwargs = {'n':2, 'm':1, 'params_interval':1080, 'device':device, 'context_dim':6, 'lstm_embed':True, 'embed_dim':42,
                     'embed_params':None, 'hyper_hidden':[32, 32], 'control_dim':1, 'lambda_hidden':[32, 32], 'V_hidden':[32, 32],
                     'Q_hidden':[32, 32], 'complex':False, 'dt':None, 'nonlinearity':torch.nn.ReLU, 'stable':True, 'B':torch.Tensor([[0.0,0.166]]),
                        'lambdas_fac':0.02, 'Q_fac':0.04,'dropout':0.3,'stationary':True, 'bias_fac':0.0}
            with open(outs_path + f"/{title}_model_kwargs_" + str(seed) + ".pkl",'wb') as f:
                pickle.dump(model_kwargs,f)
        else:
            with open(outs_path + f"/{title}_model_kwargs_" + str(seed) + ".pkl",'rb') as f:
                model_kwargs = pickle.load(f)
        model = HyperNESDE(**model_kwargs).to(device)
        if do_train:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99) # optional, set to None to disable

        # train the model:
            train_res = train_contextual_NESDE(model, dl, optim=optimizer, train_epochs=n_epochs, w_path=weights_path,
                                random_samples=5, eval_rate=1, patience=patience, sched=scheduler, verbose=1, skip_first=True, skip_id=1 if skip_first else 0)
        else:
            model.load_state_dict(torch.load(weights_path))
        # evaluate on test set:
        data = load_data(data_path, 'test', seed=None)
        for di in range(len(data)):
            new_dat1 = torch.zeros((len(data[di]['times'])+1,1))
            new_dat1[1:] = data[di]['times']
            new_dat2 = torch.zeros((len(data[di]['obs'])+1),1)
            new_dat2[1:] = data[di]['obs']
            new_dat3 = torch.ones((len(data[di]['mask'])+1,1))
            new_dat3[1:] = data[di]['mask']
            data[di]['times'] = new_dat1
            data[di]['obs'] = new_dat2
            data[di]['mask'] = new_dat3
        if test_size is None:
            test_size = len(data)
        dl = Dataloader(data, split=[0,0,test_size], batch_size=1, shuffle=False, global_shuffle=False, verbose=False)
        dl.test()
        test_nll, test_mse, test_corr, tTIMES, tNLLS, tMSES, tIDS = eval_contextual_NESDE(model, dl, skip_first=True, skip_id=1 if skip_first else 0, detailed_res=True)
        print("Test loss: NLL: " + str(np.mean(test_nll)) + ", MSE: " + str(np.mean(test_mse)))
        # save results:
        with open(res_path,'wb') as f:
            pickle.dump({'NLL':test_nll,'MSE':test_mse,'CORR':test_corr, 'tNLLS':np.array(tNLLS), 'tMSES':np.array(tMSES), 'tTIMES':np.array(tTIMES), 'tIDS':np.array(tIDS)},f)

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
    parser.add_argument('--data_path', help="pickled data path", default="./Data/Vancomycin_Dosing")
    parser.add_argument('--device', help="cuda/cpu", default="cuda")
    parser.add_argument('--outs_path', type=str, help="Where to save outputs", default="./Results_NESDE33/")
    parser.add_argument('--plot', help="to plot the whole test set",default = True)
    parser.add_argument('--plot_res', help="time resolution for plotting",default = 1.0)
    parser.add_argument('--n_seeds', help="number of seeds to run", default = 1)
    parser.add_argument('--train_size', help="train dataset size", default = 5.0/6.0)
    parser.add_argument('--valid_size', help="validation dataset size", default = 1.0/6.0)
    parser.add_argument('--test_size', help="test dataset size", default = None)
    parser.add_argument('--patience', help="stopping criteria", default = 40)
    parser.add_argument('--n_epochs', help="maximal number of epochs to train", default = 200)
    parser.add_argument('--skip_first', help="skip the first observation in the evaluation", default = False)
    parser.add_argument('--do_train', help="whether to train a model or load one", default = False)
    args = parser.parse_args()

    try:
        os.mkdir(args.outs_path)
    except:
        print("Directory exists, continues...")

    main(train_size=args.train_size,
         valid_size=args.valid_size,
         test_size=args.test_size,
         n_epochs=args.n_epochs,
         patience=args.patience,
         device=args.device,
         n_seeds=args.n_seeds,
         data_path=args.data_path,
         outs_path=args.outs_path,
         plot=args.plot,
         plot_res=args.plot_res,
         do_train=args.do_train)
