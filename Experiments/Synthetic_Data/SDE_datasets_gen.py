import numpy as np
import sdeint
import torch
import pickle
import argparse
from tqdm import tqdm

def sparse_sample(n, k):
    assert k <= n / 10
    min_id = 0
    index = []
    for i in range(k):
        max_id = int(((i + 1) / k) * n)
        index.append(np.random.randint(min_id, max_id))
        min_id = index[-1] + 1
    return torch.Tensor(index).type(torch.long)

class SynGen():
    def __init__(self,f,G):
        self.f = f
        self.G = G
    def generate(self, u, tspan, x0):
        f = lambda x, t: self.f(x,t,u)
        G = lambda x, t: self.G(x,t,u)
        return sdeint.itoint(f,G,x0,tspan)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help="which experiment. options; 'efficiency', 'ood', 'regular", type=str, default='efficiency')
    parser.add_argument('--n_datasets', help="Number of datasets to generate", type=int, default=1)
    parser.add_argument('--whole_data', help="to record the whole trajectories for plotting purposes", type=int,default=False)


    args = parser.parse_args()

    whole_data_flag = False
    n_datasets_force = False
    if args.experiment == 'efficiency':
        n_trajs = (100, 200, 400, 800, 1000)
        data_names = ('100_train','200_train','400_train','800_train','test')
        prefix = "./Data/SDE_efficiency"
        subgroups = ['complex','real']
    elif args.experiment == 'ood':
        n_trajs = (1000, 1000, 1000)
        data_names = ('train','test','ood')
        prefix = "./Data/SDE_ood"
        subgroups = ['complex', 'real']
        n_datasets_force = 5
    elif args.experiment == 'regular':
        n_trajs = (1000, 1000, 1000)
        data_names = ('train','test','ood')
        prefix = "./Data/SDE_regular"
        subgroups = ['complex']
        whole_data_flag = True


    for group, n_traj in zip(data_names, n_trajs):
        for subgroup in subgroups:
            trange = [0.0,10.0] # time range for all trajectories
            nrange = [2,6] if args.experiment == 'efficiency' else [5,8] # range for number of samples from each trajectory
            urange = [0.0,0.5] # range for action value
            init_state_mu = np.array([3.0,-3.0]) # mean of initial state distribution (normal)
            init_state_scale = np.array([1.0,1.0]) # scale of initial state distribution
            bias = np.zeros(2) # bias for the process
            random_bias = False # to insert normal noise to the bias, so each trajectory would be different
            rscale = np.array([1.0,1.0]) # scale for the normal noise added to the bias (once for each trajectory)
            u_res = 1001 # determines the resolution: dt = (trange[1]-trange[0]) / u_res
            if args.experiment == 'efficiency':
                u_corr = 0.0
            else:
                u_corr = (0.5 if group=='ood' else -0.5) # bias the control towards the current signal
            n_pp = 10 # number of different values for the action (as it is piecewise constant)
            piece_size = int(u_res / n_pp)
            # parameters of the sDE dynamics:
            # dx = [Ax + u(t)]dt + BdW
            if subgroup == 'complex':
                A = np.array([[-0.5, -2.0], [2.0, -1.0]])  # -0.75+-1.98i
            elif subgroup == 'real':
                A = np.array([[-0.5, -0.5], [-0.5, -1.0]])  # -1.3, -0.19
            B = np.diag([0.5, 0.5])  # currently, iid noise. can give any valid covariance matrix

            metadata = {'desc':'lambda t: U[int(t * ((res - 1) / (tn - t0)))]','res':u_res,'t0':trange[0],'tn':trange[1]}
            if args.experiment == 'efficiency':
                if group == 'test':
                    meta_path = [prefix + "_" + subgroup + f"_{tsize}_meta_{group}.pkl" for tsize in n_trajs[:-1]]
                else:
                    meta_path = prefix + "_" + subgroup + f"_{group.split('_')[0]}_meta_{group.split('_')[1]}.pkl"
            else:
                meta_path = prefix + "_" + subgroup + f"_meta_{group}.pkl"  # path for metadata

            if isinstance(meta_path, list):
                for mpath in meta_path:
                    with open(mpath, 'wb') as f:
                        pickle.dump(metadata, f)
            else:
                with open(meta_path, 'wb') as f:
                    pickle.dump(metadata, f)

            n_datasets = n_datasets_force if n_datasets_force else args.n_datasets
            for dset in range(n_datasets):
                if n_datasets > 1:
                    if args.experiment == 'efficiency' and group == 'test':
                        data_path = [prefix + "_" + subgroup + "_" + str(dset) + f"_{tsize}_{group}.pkl" for tsize in n_trajs[:-1]]
                    else:
                        data_path = prefix + "_" + subgroup + "_" + str(dset) + f"_{group}.pkl" # for train & evaluation purposes
                    whole_data_path = prefix + "_" + subgroup + "_" + "_whole_" + str(dset) + f"_{group}.pkl" # for plotting
                else:
                    if args.experiment == 'efficiency' and group == 'test':
                        data_path = [prefix + "_" + subgroup + f"_{tsize}_{group}.pkl" for tsize in n_trajs[:-1]]
                    else:
                        data_path = prefix + "_" + subgroup + f"_{group}.pkl" # for train & evaluation purposes
                    whole_data_path = prefix + "_" + subgroup + f"_whole_{group}.pkl" # for plotting

                def f(x, t, u):
                    return A.dot(x) + u(t)

                def G(x, t, u):
                    return B
                synthetic_generator = SynGen(f, G)

                # Generate data:
                data = []
                whole_data = []
                for s in tqdm(range(n_traj)):
                    n_smp = np.random.randint(nrange[0],nrange[1])
                    if args.experiment == 'regular':
                        smp_ids = np.arange(11) * ((u_res-1)//10)
                    else:
                        smp_ids = sparse_sample(u_res, n_smp)
                        if args.experiment == 'ood' and not any(smp_ids == 0):
                            smp_ids = torch.cat((torch.zeros(1),smp_ids),dim=0)
                        if args.experiment == 'ood' and not any(smp_ids == u_res - 1):
                            smp_ids = torch.cat((smp_ids,torch.ones(1)*(u_res - 1)),dim=0)
                        smp_ids = smp_ids.type(torch.long)
                    u_ar = np.zeros((u_res, 2))
                    for pp in range(n_pp):
                        val = np.random.normal(urange[0],urange[1])
                        u_ar[pp * piece_size:(pp+1)*piece_size,1] = val
                    tspan = np.linspace(trange[0], trange[1], u_res)
                    x0 = np.array(init_state_scale * np.random.randn(2) + init_state_mu)
                    samples_list = []
                    rbs = np.zeros(2)
                    if random_bias:
                        rbs = rscale * np.random.randn(2)
                    u = lambda t: u_ar[int(t * ((u_res-1)/ (trange[1]-trange[0])))]

                    for ij, smid in enumerate(smp_ids):
                        if smid == 0:
                            assert len(smp_ids) > 1
                            u_ar[smid:smp_ids[ij+1],1] = u_ar[smid:smp_ids[ij+1],1] + u_corr*(x0[0])
                            u = lambda t: u_ar[int(t * ((u_res-1)/ (trange[1]-trange[0])))]
                            smps = np.reshape(x0,(1,2))
                            samples_list.append(smps)
                        else:
                            start_ind = smp_ids[ij-1] if ij > 0 else 0
                            end_ind = smp_ids[ij+1] if ij < len(smp_ids)-1 else u_ar.shape[0]
                            smps = synthetic_generator.generate(u,tspan[start_ind:smid+1],x0).reshape((-1,2))
                            x0 = smps[-1]
                            if ij < len(smp_ids) - 1:
                                u_ar[smid:end_ind,1] = u_ar[smid:end_ind,1] + u_corr*(x0[0])
                                u = lambda t: u_ar[int(t * ((u_res-1)/ (trange[1]-trange[0])))]
                            if len(samples_list) > 0:
                                smps = smps[1:]
                            samples_list.append(smps)

                    samples = np.concatenate(samples_list)
                    samples = samples + bias.reshape(1,2).repeat(samples.shape[0],0)
                    if random_bias:
                        samples = samples + rbs.reshape(1,2).repeat(samples.shape[0],0)
                    samplesc = samples[:, 0]
                    maskc = np.zeros(samplesc.shape)
                    maskc[smp_ids] = 1

                    U = torch.Tensor(u_ar[:,1])
                    tspan = torch.Tensor(tspan)
                    samples = torch.Tensor(samples)
                    whole_data.append({'obs': samples.clone(), 'times': tspan.view(-1,1), 'U':U})
                    samples = samples[smp_ids][:, 0].view(-1, 1)
                    mask = torch.ones(samples.shape)
                    data.append({'obs':samples,'times':tspan[smp_ids].view(-1,1),'U':U,'mask':mask})

                # save data:
                if isinstance(data_path, list):
                    for dpath in data_path:
                        with open(dpath, 'wb') as f:
                            pickle.dump({'data': data}, f)
                else:
                    with open(data_path,'wb') as f:
                        pickle.dump({'data':data},f)
                if args.whole_data or whole_data_flag:
                    with open(whole_data_path,'wb') as f:
                        pickle.dump({'data':whole_data},f)



