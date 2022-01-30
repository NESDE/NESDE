import numpy as np
import sdeint
import torch
import pickle
import argparse

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
    parser.add_argument('--prefix', help="Prefix for generated data", default="./Data/SDE_control")
    parser.add_argument('--n_datasets', help="Number of datasets to generate", default=1)
    parser.add_argument('--whole_data', help="to record the whole trajectories for plotting purposes", default=True)
    parser.add_argument('--dynamics', help="type of process dynamics", default="complex")
    parser.add_argument('--regular', help="regular samples", default=False)

    args = parser.parse_args()

    n_trajs = (1000, 1000) # number of trajectories to sample

    for group, n_traj in zip(('train','test'), n_trajs):
        trange = [0.0,10.0] # time range for all trajectories
        nrange = [5,20] # range for number of samples from each trajectory
        urange = [0.0,5.0] # range for action value
        init_state_mu = np.array([3.0,-3.0]) # mean of initial state distribution (normal)
        init_state_scale = np.array([1.0,1.0]) # scale of initial state distribution
        bias = np.zeros(2) # bias for the process
        random_bias = False # to insert normal noise to the bias, so each trajectory would be different
        rscale = np.array([1.0,1.0]) # scale for the normal noise added to the bias (once for each trajectory)
        u_res = 10001 # determines the resolution: dt = (trange[1]-trange[0]) / u_res
        n_pp = 10 # number of different values for the action (as it is piecewise constant)
        piece_size = int(u_res / n_pp)
        # parameters of the sDE dynamics:
        # dx = [Ax + u(t)]dt + BdW
        if args.dynamics in ('complex','tanh'):
            A = np.array([[-0.5, -2.0], [2.0, -1.0]])  # -0.75+-1.98i
        elif args.dynamics == 'real':
            A = np.array([[-0.5, -0.5], [-0.5, -1.0]])  # -1.3, -0.19
        elif args.dynamics == 'imag':
            A = 0.5 * np.array([[1, -2], [2, -1]])  # +-1.71i
        B = np.diag([0.5, 0.5])  # currently, iid noise. can give any valid covariance matrix
        # define callables for the integration:
        if args.dynamics == 'tanh':
            def f(x, t, u):
                return np.tanh(A.dot(x) + u(t))
        else:
            def f(x, t, u):
                return A.dot(x) + u(t)
        def G(x, t, u):
            return B

        synthetic_generator = SynGen(f, G)

        metadata = {'desc':'lambda t: U[int(t * ((res - 1) / (tn - t0)))]','res':u_res,'t0':trange[0],'tn':trange[1]}
        meta_path = args.prefix + "_" + args.dynamics + f"_meta_{group}.pkl"  # path for metadata
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)

        for dset in range(args.n_datasets):
            if args.n_datasets > 1:
                data_path = args.prefix + "_" + args.dynamics + "_" + str(dset) + f"_{group}.pkl" # for train & evaluation purposes
                whole_data_path = args.prefix + "_" + args.dynamics + "_" + "_whole_" + str(dset) + f"_{group}.pkl" # for plotting
            else:
                data_path = args.prefix + "_" + args.dynamics + f"_{group}.pkl" # for train & evaluation purposes
                whole_data_path = args.prefix + "_" + args.dynamics + f"_whole_{group}.pkl" # for plotting


            # Generate data:
            data = []
            whole_data = []
            for s in range(n_traj):
                n_smp = np.random.randint(nrange[0],nrange[1])
                u_ar = np.zeros((u_res, 2))
                for pp in range(n_pp):
                    val = np.random.uniform(urange[0],urange[1])
                    u_ar[pp * piece_size:(pp+1)*piece_size,1] = val
                u = lambda t: u_ar[int(t * ((u_res-1)/ (trange[1]-trange[0])))]
                tspan = np.linspace(trange[0], trange[1], u_res)
                x0 = np.array(init_state_scale * np.random.randn(2) + init_state_mu)
                samples = synthetic_generator.generate(u,tspan,x0)
                samples = samples + bias.reshape(1,2).repeat(samples.shape[0],0)
                if random_bias:
                    samples = samples + rscale * np.random.randn(2).reshape(1,2).repeat(samples.shape[0],0)
                if args.regular:
                    smp_ids = np.arange(11) * ((u_res-1)//10)
                else:
                    smp_ids = sparse_sample(u_res, n_smp)
                U = torch.Tensor(u_ar[:,1])
                tspan = torch.Tensor(tspan)
                samples = torch.Tensor(samples)
                whole_data.append({'obs': samples.clone(), 'times': tspan.view(-1,1), 'U':U})
                samples = samples[smp_ids][:, 0].view(-1, 1)
                mask = torch.ones(samples.shape)
                data.append({'obs':samples,'times':tspan[smp_ids].view(-1,1),'U':U,'mask':mask})


            # save data:
            with open(data_path,'wb') as f:
                pickle.dump({'data':data},f)
            if args.whole_data:
                with open(whole_data_path,'wb') as f:
                    pickle.dump({'data':whole_data},f)



