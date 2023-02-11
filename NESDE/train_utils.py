import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd

def plot_data_NESDE(model, trj, dt, t_end=None):
    with torch.no_grad():
        NLL = None
        MSE = None
        St = None
        St_var = None
        prev_time = torch.zeros_like(trj['times'][0])
        if t_end is None:
            t_space = dt * torch.arange(((1.0/dt) * trj['times'][-1]).type(torch.long).item()+1)
        else:
            t_space = dt * torch.arange(((1.0/dt) * t_end).type(torch.long).item()+1)
        obs_list = []
        t_list = []
        pred_mu_list = []
        pred_var_list = []
        mask_list = []
        t_end_flag = False
        if (t_end is not None) and (t_end > trj['times'][-1]):
            t_end_flag = True
            ts_to_plot = torch.cat([trj['times'].view(-1),torch.ones(1)*t_end])
        else:
            ts_to_plot = trj['times']

        for i,time in enumerate(ts_to_plot):
            t_ind = t_space <= time
            t_ind = torch.logical_and(t_ind,t_space > prev_time)
            if time not in t_space[t_ind]:
                cal_times = torch.cat([t_space[t_ind].view(-1,1), time.view(-1,1)],dim=0)
            else:
                cal_times = t_space[t_ind].view(-1,1)

            if St is not None:
                St = St.repeat(cal_times.shape[0],1)
                St_var = St_var.repeat(cal_times.shape[0], 1,1)
            if model.dt is None or model.B is None:
                St, St_var = model(St, St_var, (cal_times - prev_time).to(model.device), U=trj['U'][i].view(1,-1).repeat(cal_times.shape[0],1))
            else:
                St, St_var = model(St, St_var, (cal_times - prev_time).to(model.device), U=[trj['U'] for _ in range(cal_times.shape[0])], t0_ac=[prev_time.to(model.device) for _ in range(cal_times.shape[0])])
            prev_time = time
            pred_mu_list.append(St[..., :model.m].view(-1, model.m))
            pred_var_list.append(torch.sqrt(torch.diagonal(St_var[..., :model.m, :model.m], dim1=-2, dim2=-1).view(-1, model.m)))
            if i == ts_to_plot.shape[0] - 1 and t_end_flag:
                for mt in cal_times[:-1]:
                    t_list.append(mt)
                    obs_list.append(torch.zeros_like(trj['obs'][i-1].view(1, model.m)))
                    mask_list.append(torch.zeros_like(trj['mask'][i-1].view(1, model.m)))
                t_list.append(cal_times[-1])
                obs_list.append(torch.zeros_like(trj['obs'][i-1].view(1, model.m)))
                mask_list.append(torch.zeros_like(trj['mask'][i-1].view(1, model.m)))
                continue
            else:
                for mt in cal_times[:-1]:
                    t_list.append(mt)
                    obs_list.append(torch.zeros_like(trj['obs'][i].view(1, model.m)))
                    mask_list.append(torch.zeros_like(trj['mask'][i].view(1, model.m)))
                t_list.append(cal_times[-1])
                obs_list.append(trj['obs'][i].view(1,model.m))
                mask_list.append(trj['mask'][i].view(1,model.m))
            obs_mask = trj['mask'][i].type(torch.bool)
            dim_obs = torch.sum(obs_mask)
            smp_mask = torch.zeros(2 * model.m + model.n).type(torch.bool)
            smp_mask[:model.m] = obs_mask
            smp = trj['obs'][i][obs_mask].view(-1, dim_obs).to(model.device)
            pred_mu = St[-1,..., smp_mask].view(-1, dim_obs)
            pred_var = torch.diagonal(St_var[-1], dim1=-2, dim2=-1)[..., smp_mask].view(-1, dim_obs)
            nll_loss = (1.0 / dim_obs) * torch.sum(np.log(np.sqrt(2*np.pi)) + (0.5*torch.log(torch.abs(pred_var))) + torch.square((pred_mu - smp))/(2*torch.abs(pred_var)))
            mse_loss = (1.0 / dim_obs) * torch.sum(torch.square(pred_mu - smp))
            St, St_var = model.conditional_dist(St[-1].unsqueeze(0), St_var[-1].unsqueeze(0), smp_mask, smp)
            if NLL is None:
                NLL = (1.0 / trj['times'].shape[0]) * nll_loss
                MSE = (1.0 / trj['times'].shape[0]) * mse_loss
            else:
                NLL = NLL + (1.0 / trj['times'].shape[0]) * nll_loss
                MSE = MSE + (1.0 / trj['times'].shape[0]) * mse_loss

        return {'pred_mu':pred_mu_list,'pred_var':pred_var_list,'times':t_list,'obs':obs_list,'mask':mask_list,'U':trj['U'],'t_obs':trj['times'],'nll':NLL.cpu().numpy(),'mse':MSE.cpu().numpy()}


def plot_data_contextual_NESDE(model, trj, dt, t_end=None, deterministic=False):
    with torch.no_grad():
        NLL = None
        MSE = None
        St = None
        St_var = None
        prev_time = torch.zeros_like(trj['times'][0])
        if t_end is None:
            t_space = dt * torch.arange(((1.0/dt) * trj['times'][-1]).type(torch.long).item()+1)
        else:
            t_space = dt * torch.arange(((1.0/dt) * t_end).type(torch.long).item()+1)
        model.reset_context()
        U_starts = trj['U'][:, 0]
        U_ends = trj['U'][:, 1]
        U_vals = trj['U'][:, 2]
        SI_times = trj['SI'][1:, 0]
        SI_vals = trj['SI'][1:, 1:]
        model.set_context(trj['SI'][0, 1:].to(model.device))
        U_for_plot = torch.ones_like(t_space)
        for ii in range(trj['U'].shape[0]):
            t_idx = t_space >= U_starts[ii]
            t_idx = torch.logical_and(t_idx,t_space < U_ends[ii])
            U_for_plot[t_idx] = U_vals[ii] * U_for_plot[t_idx]
        obs_list = []
        t_list = []
        pred_mu_list = []
        pred_var_list = []
        mask_list = []
        t_end_flag = False
        if (t_end is not None) and (t_end > trj['times'][-1]):
            t_end_flag = True
            ts_to_plot = torch.cat([trj['times'].view(-1),torch.ones(1)*t_end])
        else:
            ts_to_plot = trj['times']

        for i,time in enumerate(ts_to_plot):
            t_ind = t_space <= time
            t_ind = torch.logical_and(t_ind,t_space > prev_time)
            if time not in t_space[t_ind]:
                cal_times = torch.cat([t_space[t_ind].view(-1,1), time.view(-1,1)],dim=0)
            else:
                cal_times = t_space[t_ind].view(-1,1)
            if isinstance(prev_time,torch.Tensor):
                prev_time = prev_time.item()
            U_idx = U_starts <= time
            U_idx = torch.logical_and(U_idx, U_ends > prev_time)
            U_start = torch.clip(U_starts[U_idx], min=prev_time)
            U_end = torch.clip(U_ends[U_idx], max=time)
            U_val = U_vals[U_idx].to(model.device)
            SI_idx = SI_times >= prev_time
            SI_idx = torch.logical_and(SI_idx, SI_times < time)
            SI_time = SI_times[SI_idx]
            SI_val = SI_vals[SI_idx]
            SI_idx = torch.ones_like(SI_time).type(torch.bool)
            St_list = []
            St_var_list = []
            cal_t_idx = torch.ones_like(cal_times).type(torch.bool)
            for act in range(torch.sum(U_idx)):
                if U_start[act] > prev_time:
                    valid_SI_idx = SI_time[SI_idx] < U_start[act]
                    if torch.any(valid_SI_idx):
                        for jj, si in enumerate(SI_val[SI_idx][valid_SI_idx]):
                            t_idx = torch.logical_and(cal_t_idx,cal_times <= SI_time[SI_idx][valid_SI_idx][jj])
                            model.set_context(si.to(model.device))
                            if torch.any(t_idx):
                                if St is not None:
                                    St = St.repeat(cal_times[t_idx].shape[0], 1)
                                    St_var = St_var.repeat(cal_times[t_idx].shape[0], 1, 1)
                                St, St_var = model(St, St_var, (
                                        cal_times[t_idx] - prev_time).view(-1, 1).to(model.device),
                                               U=torch.zeros_like(U_val[act].view(1, -1)).repeat(cal_times[t_idx].shape[0],1))
                                prev_time = cal_times[t_idx][-1]
                                cal_t_idx = torch.logical_and(cal_t_idx, cal_times > prev_time)
                                St_list.append(St)
                                St_var_list.append(St_var)
                                St = St[-1].unsqueeze(0)
                                St_var = St_var[-1].unsqueeze(0)
                    t_idx = torch.logical_and(cal_t_idx, cal_times <= U_start[act])
                    if torch.any(t_idx):
                        if St is not None:
                            St = St.repeat(cal_times[t_idx].shape[0], 1)
                            St_var = St_var.repeat(cal_times[t_idx].shape[0], 1, 1)
                        St, St_var = model(St, St_var, (cal_times[t_idx] - prev_time).view(-1, 1).to(model.device),
                                        U=torch.zeros_like(U_val[act].view(1, -1)).repeat(cal_times[t_idx].shape[0],1))
                        SI_idx = torch.logical_and(SI_idx, SI_time >= U_start[act])
                        prev_time = cal_times[t_idx][-1]
                        cal_t_idx = torch.logical_and(cal_t_idx, cal_times > prev_time)
                        St_list.append(St)
                        St_var_list.append(St_var)
                        St = St[-1].unsqueeze(0)
                        St_var = St_var[-1].unsqueeze(0)
                    if not torch.any(t_idx) or U_start[act] > prev_time:
                        St, St_var = model(St, St_var, (U_start[act] - prev_time).view(-1, 1).to(model.device),
                                        U=torch.zeros_like(U_val[act].view(1, -1)))
                        prev_time = U_start[act]
                        St = St[-1].unsqueeze(0)
                        St_var = St_var[-1].unsqueeze(0)
                valid_SI_idx = SI_time[SI_idx] < U_end[act]
                if torch.any(valid_SI_idx):
                    for jj, si in enumerate(SI_val[SI_idx][valid_SI_idx]):
                        t_idx = torch.logical_and(cal_t_idx, cal_times <= SI_time[SI_idx][valid_SI_idx][jj])
                        model.set_context(si.to(model.device))
                        if torch.any(t_idx):
                            if St is not None:
                                St = St.repeat(cal_times[t_idx].shape[0], 1)
                                St_var = St_var.repeat(cal_times[t_idx].shape[0], 1, 1)
                            St, St_var = model(St, St_var,
                                           (cal_times[t_idx] - prev_time).view(-1,1).to(model.device),
                                           U=U_val[act].view(1, -1).repeat(cal_times[t_idx].shape[0],1))
                            prev_time = cal_times[t_idx][-1]
                            cal_t_idx = torch.logical_and(cal_t_idx, cal_times > prev_time)
                            St_list.append(St)
                            St_var_list.append(St_var)
                            St = St[-1].unsqueeze(0)
                            St_var = St_var[-1].unsqueeze(0)
                t_idx = torch.logical_and(cal_t_idx, cal_times <= U_end[act])
                if torch.any(t_idx):
                    if St is not None:
                        St = St.repeat(cal_times[t_idx].shape[0], 1)
                        St_var = St_var.repeat(cal_times[t_idx].shape[0], 1, 1)
                    St, St_var = model(St, St_var, (cal_times[t_idx] - prev_time).view(-1, 1).to(model.device), U=U_val[act].view(1, -1).repeat(cal_times[t_idx].shape[0],1))
                    SI_idx = torch.logical_and(SI_idx, SI_time >= U_end[act])
                    prev_time = cal_times[t_idx][-1]
                    cal_t_idx = torch.logical_and(cal_t_idx, cal_times > prev_time)
                    St_list.append(St)
                    St_var_list.append(St_var)
                    St = St[-1].unsqueeze(0)
                    St_var = St_var[-1].unsqueeze(0)
                if not torch.any(t_idx) or U_end[act] > prev_time:
                    St, St_var = model(St, St_var, (U_end[act] - prev_time).view(-1, 1).to(model.device), U=U_val[act].view(1, -1))
                    prev_time = U_end[act]
                    St = St[-1].unsqueeze(0)
                    St_var = St_var[-1].unsqueeze(0)
            if prev_time < cal_times[-1]:
                valid_SI_idx = SI_time[SI_idx] < cal_times[-1]
                if torch.any(valid_SI_idx):
                    for jj, si in enumerate(SI_val[SI_idx][valid_SI_idx]):
                        t_idx = torch.logical_and(cal_t_idx, cal_times <= SI_time[SI_idx][valid_SI_idx][jj])
                        model.set_context(si.to(model.device))
                        if torch.any(t_idx):
                            if St is not None:
                                St = St.repeat(cal_times[t_idx].shape[0], 1)
                                St_var = St_var.repeat(cal_times[t_idx].shape[0], 1, 1)
                            St, St_var = model(St, St_var,
                                           (cal_times[t_idx] - prev_time).view(-1,1).to(model.device),
                                           U=torch.zeros_like(trj['U'][0, 2].view(1, -1)).to(model.device).repeat(cal_times[t_idx].shape[0],1))
                            prev_time = cal_times[t_idx][-1]
                            cal_t_idx = torch.logical_and(cal_t_idx, cal_times > prev_time)
                            St_list.append(St)
                            St_var_list.append(St_var)
                            St = St[-1].unsqueeze(0)
                            St_var = St_var[-1].unsqueeze(0)
                if torch.any(cal_t_idx):
                    if St is not None:
                        St = St.repeat(cal_times[cal_t_idx].shape[0], 1)
                        St_var = St_var.repeat(cal_times[cal_t_idx].shape[0], 1, 1)
                    St, St_var = model(St, St_var, (cal_times[cal_t_idx] - prev_time).view(-1, 1).to(model.device),
                                   U=torch.zeros_like(trj['U'][0, 2].view(1, -1)).to(model.device).repeat(cal_times[cal_t_idx].shape[0],1))
                    St_list.append(St)
                    St_var_list.append(St_var)
            if len(St_list) <= 0:
                St = None
            else:
                St = torch.cat(St_list,dim=0)
                St_var = torch.cat(St_var_list,dim=0)
            if St is None:
                St, St_var = model.get_prior()

            prev_time = time
            pred_mu_list.append(St[..., :model.m].view(-1, model.m))
            pred_var_list.append(torch.sqrt(torch.diagonal(St_var[..., :model.m, :model.m], dim1=-2, dim2=-1).view(-1, model.m)))
            if i == ts_to_plot.shape[0] - 1 and t_end_flag:
                for mt in cal_times[:-1]:
                    t_list.append(mt)
                    obs_list.append(torch.zeros_like(trj['obs'][i-1].view(1, model.m)))
                    mask_list.append(torch.zeros_like(trj['mask'][i-1].view(1, model.m)))
                t_list.append(cal_times[-1])
                obs_list.append(torch.zeros_like(trj['obs'][i-1].view(1, model.m)))
                mask_list.append(torch.zeros_like(trj['mask'][i-1].view(1, model.m)))
                continue
            else:
                for mt in cal_times[:-1]:
                    t_list.append(mt)
                    obs_list.append(torch.zeros_like(trj['obs'][i].view(1, model.m)))
                    mask_list.append(torch.zeros_like(trj['mask'][i].view(1, model.m)))
                t_list.append(cal_times[-1])
                obs_list.append(trj['obs'][i].view(1,model.m))
                mask_list.append(trj['mask'][i].view(1,model.m))
            obs_mask = trj['mask'][i].type(torch.bool)
            dim_obs = torch.sum(obs_mask)
            smp_mask = torch.zeros(2 * model.m + model.n).type(torch.bool)
            smp_mask[:model.m] = obs_mask
            smp = trj['obs'][i][obs_mask].view(-1, dim_obs).to(model.device)
            if deterministic:
                pred_mu = St[-1].view(-1, dim_obs)
                mse_loss = (1.0 / dim_obs) * torch.sum(torch.square(pred_mu - smp))
                nll_loss = torch.zeros_like(mse_loss)
            else:
                pred_mu = St[-1, ..., smp_mask].view(-1, dim_obs)
                pred_var = torch.diagonal(St_var[-1], dim1=-2, dim2=-1)[..., smp_mask].view(-1, dim_obs)
                mse_loss = (1.0 / dim_obs) * torch.sum(torch.square(pred_mu - smp))
                nll_loss = (1.0 / dim_obs) * torch.sum(np.log(np.sqrt(2*np.pi)) + (0.5*torch.log(torch.abs(pred_var))) + torch.square((pred_mu - smp))/(2*torch.abs(pred_var)))
            if not deterministic:
                St, St_var = model.conditional_dist(St[-1].unsqueeze(0), St_var[-1].unsqueeze(0), smp_mask, smp)
            if NLL is None:
                NLL = (1.0 / trj['times'].shape[0]) * nll_loss
                MSE = (1.0 / trj['times'].shape[0]) * mse_loss
            else:
                NLL = NLL + (1.0 / trj['times'].shape[0]) * nll_loss
                MSE = MSE + (1.0 / trj['times'].shape[0]) * mse_loss
        U = trj['U_org'].cpu().numpy()
        if U[0,0] > 0:
            U = np.concatenate((np.zeros(U[0].shape)[None,...],U),axis=0)
        if U[-1,0] < t_list[-1]:
            ulast = np.zeros(U[0].shape)[None,...]
            ulast[0,0] = U[-1,1]
            U = np.concatenate((U,ulast),axis=0)
            ulast[0,0] = t_list[-1]
            U = np.concatenate((U,ulast),axis=0)

        return {'pred_mu':pred_mu_list,'pred_var':pred_var_list,'times':t_list,'obs':obs_list,'mask':mask_list,'U':U,'t_obs':trj['times'],'nll':NLL.cpu().numpy(),'mse':MSE.cpu().numpy()}

def plot_(times, preds, pred_vars, obs, NLL, MSE, control, whole_data=None, pw_control=False):
    if control is not None:
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax0 = plt.subplot(gs[0])
        for m in range(len(preds)):
            p = ax0.plot(times[-1], preds[m], label="model output - " + str(m))
            ax0.fill_between(times[-1], preds[m] - pred_vars[m], preds[m] + pred_vars[m], color=p[0].get_color(), alpha=0.1)
            if whole_data is not None:
                ax0.plot(whole_data['times'].cpu().numpy(), whole_data['obs'][...,m].cpu().numpy(),':', label='whole - ' + str(m), alpha=0.5, color=p[0].get_color())
            ax0.scatter(times[m], obs[m], label="samples - " + str(m),c='r')
            ax1 = plt.subplot(gs[1], sharex=ax0)
        if pw_control:
            ax1.step(control[:,0], control[:,2], color='g')
        else:
            ax1.plot(times[-1], control, color='g')
        ax0.set_ylabel("Y")
        ax1.set_ylabel("Control")
        ax1.set_xlabel("Time")
        ax0.set_title("NLL: " + str(NLL) + ", MSE: " + str(MSE))
        plt.setp(ax0.get_xticklabels(), visible=False)
        yticks = ax1.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)
        ax0.legend()
    else:
        fig = plt.figure()
        for m in range(len(preds)):
            p = plt.plot(times[-1], preds[m], label="model output - " + str(m))
            plt.fill_between(times[-1], preds[m] - pred_vars[m], preds[m] + pred_vars[m], color=p[0].get_color(), alpha=0.1)
            if whole_data is not None:
                plt.plot(whole_data['times'], whole_data['obs'][m], label='whole - ' + str(m), alpha=0.5)
            plt.scatter(times[m], obs[m], label="samples - " + str(m))
        plt.ylabel("Y")
        plt.xlabel("Time")
        plt.title("NLL: " + str(NLL) + ", MSE: " + str(MSE))
        plt.legend()

    return fig


def plot_traj(plot_data, control, m, path, whole_data=None,pw_control=False):
    preds = torch.cat(plot_data['pred_mu'],dim=0).cpu().numpy()
    pred_vars = torch.cat(plot_data['pred_var'],dim=0).cpu().numpy()
    times = torch.cat(plot_data['times'],dim=0).cpu().numpy()
    obs = torch.cat(plot_data['obs'],dim=0).cpu().numpy()
    mask = torch.cat(plot_data['mask'],dim=0).type(torch.bool).cpu().numpy()
    preds = [preds[:,i].reshape(-1) for i in range(m)]
    pred_vars = [pred_vars[:,i].reshape(-1) for i in range(m)]
    mask = [mask[:,i].reshape(-1) for i in range(m)]
    obs = [obs[:,i][mask[i]].reshape(-1) for i in range(m)]
    ttimes = [times[mask[i]].reshape(-1) for i in range(m)]
    ttimes.append(times.reshape(-1))

    fig = plot_(times=ttimes,preds=preds,pred_vars=pred_vars,obs=obs,NLL=plot_data['nll'],MSE=plot_data['mse'],control=control,whole_data=whole_data,pw_control=pw_control)
    fig.savefig(path)


def plot_NESDE(model, dl, dt, path, whole_data=None,plot_control=False,pw_control=False,contextual=False, deterministic=False):
    epoch_done = False
    n_batches = 0
    whole_trj = None
    t_end = None
    while not epoch_done:
        batch, epoch_done, ids = dl.get_batch(return_ids=True)
        for i, trj in enumerate(batch):
            if whole_data is not None:
                t_end = whole_data[ids[i]]['times'][-1]
            if contextual:
                trj_res = plot_data_contextual_NESDE(model, trj, dt, t_end=t_end, deterministic=deterministic)
            else:
                trj_res = plot_data_NESDE(model,trj,dt,t_end=t_end)
            control = None
            if model.B is not None and model.dt is not None:
                control = torch.cat([trj_res['U'](tt).view(1) for tt in trj_res['times']],dim=0).cpu().numpy()
            if plot_control:
                control = trj_res['U']
            if whole_data is not None:
                whole_trj = whole_data[ids[i]]
            plot_traj(trj_res,control=control,m=model.m, path=path + '/trj_' + str(i + n_batches * dl.batch_size) + '.png',whole_data=whole_trj, pw_control=pw_control)
        n_batches += 1


def eval_NESDE(model, dl, Tmin=None, n_samp_per_traj=None, verbose=0, detailed_res=False, skip_first=False, deterministic=False, irreg_lstm=False):
    with torch.no_grad():
        t0 = time.time()
        model.eval()
        NLLs, MSEs, CORRs, lambdas = [], [], [], []
        tTIMEs, tNLLs, tMSEs = [], [], []
        n_batches = 0
        epoch_done = False
        while not epoch_done:
            batch, epoch_done = dl.get_batch()
            n_batches += 1
            b_size = 0
            for trj in batch:
                b_size += 1
                traj_id = torch.arange(trj['times'].shape[0]).type(torch.long)
                St = None
                St_var = None
                preds_mu = []
                n_obs = 0
                n_obs_before_Tmin = 0
                trj_nll = 0
                trj_mse = 0
                prev_time = 0
                if irreg_lstm:
                    u_change = trj['U'].view(-1)
                    u_change = (u_change[1:] - u_change[:-1]) != 0
                    u_change = torch.cat((torch.ones(1).type(torch.bool),u_change))
                    uc_ids = torch.arange(len(u_change))
                for id in (traj_id):
                    ttimes = trj['times'][id].to(model.device)
                    curr_time = np.round(ttimes.max().item(),4)
                    if n_samp_per_traj is not None and n_obs >= n_samp_per_traj:
                        break

                    if irreg_lstm:
                        ptime = trj['times'][id].item()
                        pid = (ptime / model.dtr).item()
                        pid = int(pid+0.5)
                        cid = (prev_time / model.dtr).item()
                        cid = int(cid+0.5)
                        puid = cid
                        if cid == pid:
                            St, St_var = model(St, St_var, ttimes - prev_time, U=torch.zeros_like(trj['U'][puid]))
                        if torch.any(u_change[cid:pid]):
                            uids = uc_ids[cid+1:pid][u_change[cid+1:pid]]
                            for uid in uids:
                                St, St_var = model(St, St_var, uid*model.dtr - prev_time, U=trj['U'][puid])
                                puid = uid
                                prev_time = uid * model.dtr
                        if puid < pid:
                            St, St_var = model(St, St_var, ttimes - prev_time, U=trj['U'][puid])
                    elif model.dt is None or model.B is None:
                        St, St_var = model(St, St_var, ttimes - prev_time, U=trj['U'][id])
                    else:
                        St, St_var = model(St, St_var, ttimes - prev_time, U=[trj['U']], t0_ac=[prev_time])
                    prev_time = ttimes.cpu().item()

                    obs_mask = trj['mask'][id].type(torch.bool)
                    dim_obs = torch.sum(obs_mask)
                    smp_mask = torch.zeros(2 * model.m + model.n).type(torch.bool)
                    smp_mask[:model.m] = obs_mask
                    smp = trj['obs'][id][obs_mask].view(-1, dim_obs).to(model.device)
                    if deterministic:
                        pred_mu = St.view(-1, dim_obs)

                        preds_mu.append(pred_mu.item())
                        pred_var = torch.zeros_like(pred_mu)
                        mse_loss = (1.0 / dim_obs) * torch.sum(torch.square(pred_mu - smp))
                        nll_loss = torch.zeros_like(mse_loss)
                    else:
                        pred_mu = St[..., smp_mask].view(-1, dim_obs)
                        preds_mu.append(pred_mu[0,0].item())
                        pred_var = torch.diagonal(St_var, dim1=-2, dim2=-1)[..., smp_mask].view(-1, dim_obs)
                        nll_loss = (1.0 / dim_obs) * torch.sum(np.log(np.sqrt(2 * np.pi)) + (0.5 * torch.log(torch.abs(pred_var))) + torch.square((pred_mu - smp)) / (2 * torch.abs(pred_var)))
                        mse_loss = (1.0 / dim_obs) * torch.sum(torch.square(pred_mu - smp))
                    if not deterministic:
                        if not (Tmin is not None and n_samp_per_traj is not None and curr_time > Tmin):
                            St, St_var = model.conditional_dist(St, St_var, smp_mask, smp)

                    if Tmin is None or curr_time <= Tmin:
                        n_obs_before_Tmin += 1
                    if Tmin is None or curr_time > Tmin:
                        if n_obs_before_Tmin == 0:
                            # if Tmin!=None, only consider trajs with at least 1 sample before Tmin
                            break
                        n_obs += 1
                        if Tmin is not None or (n_obs > 1 or not skip_first):
                            trj_nll += nll_loss.item()
                            trj_mse += mse_loss.item()
                            if detailed_res:
                                tTIMEs.append(curr_time)
                                tNLLs.append(nll_loss.item())
                                tMSEs.append(mse_loss.item())

                if n_obs > 0:
                    reduce_count = skip_first and Tmin is None
                    NLLs.append(trj_nll / (n_obs-reduce_count))
                    MSEs.append(trj_mse / (n_obs-reduce_count))
                    CORRs.append(np.corrcoef((preds_mu, [trj['obs'][iidd][0] for iidd in traj_id]))[0,1] if n_obs>1 else None)
                    if not deterministic:
                        lambdas.append(model.curr_params['lambda'].detach().cpu().numpy())

        if verbose >= 1:
            print(f'Test:\tNLL={np.mean(NLLs):.4f}\tMSE={np.mean(MSEs):.4f}\t({time.time()-t0:.0f}s)')

        if detailed_res:
            return NLLs, MSEs, CORRs, tTIMEs, tNLLs, tMSEs, lambdas
        return NLLs, MSEs, CORRs


def train_NESDE(model, dl, optim, train_epochs, w_path, random_samples=False, eval_rate=1, patience=200,
                sched=None, train_nll=True, valid_by_nll=True, Tmin=None, n_samp_per_traj=None, verbose=1, log_rate=10, meta_nums=None, skip_first=False, deterministic=False, irreg_lstm=False):
    iterations, epochs, groups, NLLs, MSEs = [], [], [], [], []
    iter = -1
    pat_count = 0
    best_valid_loss = np.inf
    t0 = time.time()
    for t in range(train_epochs):
        dl.train()
        model.train()
        n_batches = 0
        mean_loss = 0
        mean_mseloss = 0
        epoch_done = False
        while not epoch_done:
            iter += 1
            batch, epoch_done = dl.get_batch()
            n_batches += 1
            NLL = None
            optim.zero_grad()
            b_size = 0
            for trj in batch:
                b_size += 1
                if random_samples:
                    traj_id, _ = torch.sort(torch.randperm(trj['times'].shape[0])[:random_samples])
                else:
                    traj_id = torch.arange(trj['times'].shape[0]).type(torch.long)
                St = None
                St_var = None
                prev_time = 0
                if irreg_lstm:
                    u_change = trj['U'].view(-1)
                    u_change = (u_change[1:] - u_change[:-1]) != 0
                    u_change = torch.cat((torch.ones(1).type(torch.bool),u_change))
                    uc_ids = torch.arange(len(u_change))
                for id in (traj_id):
                    ttimes = trj['times'][id].to(model.device)
                    if irreg_lstm:
                        ptime = trj['times'][id].item()
                        pid = (ptime / model.dtr).item()
                        pid = int(pid + 0.5)
                        cid = (prev_time / model.dtr).item()
                        cid = int(cid + 0.5)
                        puid = cid
                        if cid == pid:
                            St, St_var = model(St, St_var, ttimes - prev_time, U=torch.zeros_like(trj['U'][puid]))
                        if torch.any(u_change[cid:pid]):
                            uids = uc_ids[cid + 1:pid][u_change[cid + 1:pid]]
                            for uid in uids:
                                St, St_var = model(St, St_var, uid * model.dtr - prev_time, U=trj['U'][puid])
                                puid = uid
                                prev_time = uid * model.dtr
                        if puid < pid:
                            St, St_var = model(St, St_var, ttimes - prev_time, U=trj['U'][puid])
                    elif model.dt is None or model.B is None:
                        St, St_var = model(St, St_var, ttimes - prev_time, U=trj['U'][id])
                    else:
                        St, St_var = model(St, St_var, ttimes - prev_time, U=[trj['U']], t0_ac=[prev_time])
                    prev_time = ttimes.cpu().item()

                    obs_mask = trj['mask'][id].type(torch.bool)
                    dim_obs = torch.sum(obs_mask)
                    smp_mask = torch.zeros(2 * model.m + model.n).type(torch.bool)
                    smp_mask[:model.m] = obs_mask
                    smp = trj['obs'][id][obs_mask].view(-1,dim_obs).to(model.device)
                    if meta_nums is not None:
                        smp = (smp - meta_nums['mean_1']) / meta_nums['std_1']
                    if deterministic:
                        pred_mu = St.view(-1, dim_obs)
                        mseloss = model.loss_fn(pred_mu, smp)
                        loss = torch.zeros_like(mseloss)
                    else:

                        pred_mu = St[...,smp_mask].view(-1,dim_obs)
                        pred_var = torch.abs(torch.diagonal(St_var,dim1=-2,dim2=-1)[...,smp_mask].view(-1,dim_obs))
                        loss = model.loss_fn(pred_mu, smp, pred_var)
                        mseloss = (1.0 / dim_obs) * torch.sum(torch.square(pred_mu - smp))

                    if not train_nll:
                        mse_loss = (1.0 / dim_obs) * torch.sum(torch.square(pred_mu - smp))
                        loss = loss + mse_loss
                    if torch.isnan(loss):
                        loss = torch.zeros(1,device=model.device,requires_grad=True)
                    if not deterministic:
                        St, St_var = model.conditional_dist(St, St_var, smp_mask, smp)

                    if NLL is None:
                        NLL = (1.0 / len(traj_id)) * loss
                        MSE = (1.0 / len(traj_id)) * mseloss
                    else:
                        NLL = NLL + (1.0 / len(traj_id)) * loss
                        MSE = MSE + (1.0 / len(traj_id)) * mseloss

            NLL = (1.0 / b_size) * NLL
            MSE = (1.0 / b_size) * MSE
            mean_loss = mean_loss + NLL.detach()
            mean_mseloss = mean_mseloss + MSE.detach()
            if deterministic:
                MSE.backward()
            else:
                NLL.backward()
            optim.step()

            epochs.append(t)
            iterations.append(iter)
            groups.append('train')
            NLLs.append(NLL.item())
            MSEs.append(None)
        mean_loss = (1.0 / n_batches) * mean_loss
        if verbose >= 2:
            print("Train loss at epoch " + str(t + 1) + ": " + str(mean_loss.item()))
        if (t + 1) % eval_rate == 0:
            dl.valid()
            valid_nll, valid_mse, _ = eval_NESDE(model, dl, Tmin=Tmin, n_samp_per_traj=n_samp_per_traj,skip_first=skip_first, deterministic=deterministic)
            valid_nll, valid_mse = np.mean(valid_nll), np.mean(valid_mse)

            epochs.append(t)
            iterations.append(iter)
            groups.append('valid')
            NLLs.append(valid_nll)
            MSEs.append(valid_mse)
            if verbose >= 1 and (t%(eval_rate*log_rate)) == 0:
                print(f"[validation] epoch {t:02d}:\tNLL={valid_nll:.4f}\tMSE={valid_mse:.4f}\t({time.time()-t0:.0f}s)")

            valid_loss = valid_nll if valid_by_nll else valid_mse
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                pat_count = 0
                torch.save(model.state_dict(), w_path)
            else:
                pat_count += 1

            if pat_count > patience:
                print("Met stopping criteria after " + str(t + 1) + " epochs. Stopping!")
                model.load_state_dict(torch.load(w_path))
                return train_res_summary(epochs, iterations, groups, NLLs, MSEs)
        if sched is not None:
            sched.step()

    model.load_state_dict(torch.load(w_path))
    return train_res_summary(epochs, iterations, groups, NLLs, MSEs)


def eval_contextual_NESDE(model, dl, deterministic=False, save_data=False, skip_first=False, detailed_res=False):
    with torch.no_grad():
        model.eval()
        NLLs, MSEs, CORRs = [], [], []
        tNLLs, tMSEs, tTIMEs, tIDS = [], [], [], []
        n_batches = 0
        if save_data:
            dlist = []
        epoch_done = False
        b_id = 0
        while not epoch_done:
            batch, epoch_done = dl.get_batch()
            n_batches += 1
            b_size = 0
            for trj in batch:
                b_size += 1
                traj_id = torch.arange(trj['times'].shape[0]).type(torch.long)
                St = None
                St_var = None
                preds_mu = []
                trj_nll = 0
                trj_mse = 0
                prev_time = 0
                if save_data:
                    ddic = {'obs':trj['obs'].cpu().numpy(),'times':trj['times'].cpu().numpy()}
                    preds_to_save = []
                    vars_to_save = []
                model.reset_context()
                U_starts = trj['U'][:, 0]
                U_ends = trj['U'][:, 1]
                U_vals = trj['U'][:, 2]
                SI_times = trj['SI'][1:, 0]
                SI_vals = trj['SI'][1:, 1:]
                model.set_context(trj['SI'][0, 1:].to(model.device))
                for id in (traj_id):
                    ttimes = trj['times'][id].to(model.device)
                    curr_time = np.round(ttimes.max().item(),4)
                    if isinstance(prev_time,torch.Tensor):
                        prev_time = prev_time.item()
                    U_idx = U_starts <= trj['times'][id]
                    U_idx = torch.logical_and(U_idx, U_ends > prev_time)
                    U_start = torch.clip(U_starts[U_idx],min=torch.Tensor([prev_time]).item())
                    U_end = torch.clip(U_ends[U_idx],max=ttimes.item())
                    U_val = U_vals[U_idx].to(model.device)
                    SI_idx = SI_times >= prev_time
                    SI_idx = torch.logical_and(SI_idx,SI_times < trj['times'][id])
                    SI_time = SI_times[SI_idx]
                    SI_val = SI_vals[SI_idx]
                    SI_idx = torch.ones_like(SI_time).type(torch.bool)
                    for act in range(torch.sum(U_idx)):
                        if U_start[act] > prev_time:
                            valid_SI_idx = SI_time[SI_idx] < U_start[act]
                            if torch.any(valid_SI_idx):
                                for jj, si in enumerate(SI_val[SI_idx][valid_SI_idx]):
                                    model.set_context(si.to(model.device))
                                    St, St_var = model(St, St_var, (SI_time[SI_idx][valid_SI_idx][jj].to(
                                        model.device) - prev_time).view(-1, 1),
                                                       U=torch.zeros_like(U_val[act].view(1, -1)))
                                    prev_time = SI_time[SI_idx][valid_SI_idx][jj].to(model.device)
                            St, St_var = model(St, St_var, (U_start[act].to(model.device) - prev_time).view(-1, 1),
                                               U=torch.zeros_like(U_val[act].view(1, -1)))
                            SI_idx = torch.logical_and(SI_idx,SI_time >= U_start[act])
                            prev_time = U_start[act]
                        valid_SI_idx = SI_time[SI_idx] < U_end[act]
                        if torch.any(valid_SI_idx):
                            for jj, si in enumerate(SI_val[SI_idx][valid_SI_idx]):
                                model.set_context(si.to(model.device))
                                St, St_var = model(St, St_var, (
                                        SI_time[SI_idx][valid_SI_idx][jj].to(model.device) - prev_time).view(-1,
                                                                                                             1),
                                                   U=U_val[act].view(1, -1))
                                prev_time = SI_time[SI_idx][valid_SI_idx][jj].to(model.device)
                        St, St_var = model(St, St_var, (U_end[act].to(model.device) - prev_time).view(-1,1), U=U_val[act].view(1,-1))
                        SI_idx = torch.logical_and(SI_idx, SI_time >= U_end[act])
                        prev_time = U_end[act]
                    if prev_time < trj['times'][id]:
                        valid_SI_idx = SI_time[SI_idx] < trj['times'][id]
                        if torch.any(valid_SI_idx):
                            for jj, si in enumerate(SI_val[SI_idx][valid_SI_idx]):
                                model.set_context(si.to(model.device))
                                St, St_var = model(St, St_var, (
                                        SI_time[SI_idx][valid_SI_idx][jj].to(model.device) - prev_time).view(-1,
                                                                                                             1),
                                                   U=torch.zeros_like(trj['U'][0, 2].view(1, -1)).to(model.device))
                                prev_time = SI_time[SI_idx][valid_SI_idx][jj].to(model.device)
                        St, St_var = model(St, St_var, (ttimes - prev_time).view(-1, 1),
                                           U=torch.zeros_like(trj['U'][0,2].view(1, -1)).to(model.device))
                    if St is None:
                        St, St_var = model.get_prior()

                    prev_time = ttimes

                    obs_mask = trj['mask'][id].type(torch.bool)
                    dim_obs = torch.sum(obs_mask)
                    smp_mask = torch.zeros(2 * model.m + model.n).type(torch.bool)
                    smp_mask[:model.m] = obs_mask
                    smp = trj['obs'][id][obs_mask].view(-1, dim_obs).to(model.device)
                    if deterministic:
                        pred_mu = St.view(-1, dim_obs)

                        preds_mu.append(pred_mu.item())
                        pred_var = torch.zeros_like(pred_mu)
                        if save_data:
                            preds_to_save.append(pred_mu[0, 0].item())
                            vars_to_save.append(pred_var.item())
                        mse_loss = (1.0 / dim_obs) * torch.sum(torch.square(pred_mu - smp))
                        nll_loss = torch.zeros_like(mse_loss)
                    else:
                        pred_mu = St[..., smp_mask].view(-1, dim_obs)

                        preds_mu.append(pred_mu[0, 0].item())
                        pred_var = torch.diagonal(St_var, dim1=-2, dim2=-1)[..., smp_mask].view(-1, dim_obs)
                        if save_data:
                            preds_to_save.append(pred_mu[0, 0].item())
                            vars_to_save.append(pred_var.item())
                        mse_loss = (1.0 / dim_obs) * torch.sum(torch.square(pred_mu - smp))
                        nll_loss = (1.0 / dim_obs) * torch.sum(np.log(np.sqrt(2 * np.pi)) + (0.5 * torch.log(torch.abs(pred_var))) + torch.square((pred_mu - smp)) / (2 * torch.abs(pred_var)))
                    if not deterministic:
                        St, St_var = model.conditional_dist(St, St_var, smp_mask, smp)
                    if not skip_first:
                        trj_nll += nll_loss.item()
                        trj_mse += mse_loss.item()
                    if detailed_res:
                        tTIMEs.append(curr_time)
                        tNLLs.append(nll_loss.item())
                        tMSEs.append(mse_loss.item())
                        tIDS.append(b_id)
                b_id += 1
                NLLs.append(trj_nll / (len(traj_id)-skip_first))
                MSEs.append(trj_mse / (len(traj_id)-skip_first))
                CORRs.append(np.corrcoef((preds_mu[skip_first:], [o[0] for o in trj['obs'][skip_first:]]))[0,1])
                if save_data:
                    ddic['mu'] = np.array(preds_to_save)
                    ddic['var'] = np.array(vars_to_save)
                    dlist.append(ddic)

        if detailed_res:
            return NLLs, MSEs, CORRs, tTIMEs, tNLLs, tMSEs, tIDS
        if save_data:
            return NLLs, MSEs, CORRs, dlist
        return NLLs, MSEs, CORRs


def train_contextual_NESDE(model, dl, optim, train_epochs, w_path, random_samples=False, eval_rate=1, patience=200,
                           sched=None, verbose=1,deterministic=False, skip_first=False):
    iterations, epochs, groups, NLLs, MSEs = [], [], [], [], []
    iter = -1
    pat_count = 0
    best_valid_loss = np.inf
    t0 = time.time()
    mse_fn = torch.nn.MSELoss()
    for t in range(train_epochs):
        dl.train()
        model.train()
        n_batches = 0
        mean_loss = 0
        mean_msl = 0
        epoch_done = False
        while not epoch_done:
            iter += 1
            batch, epoch_done = dl.get_batch()
            n_batches += 1
            NLL = None
            MSE = None
            optim.zero_grad()
            b_size = 0
            for trj in batch:
                b_size += 1
                if random_samples:
                    traj_id, _ = torch.sort(torch.randperm(trj['times'].shape[0])[:random_samples])
                else:
                    traj_id = torch.arange(trj['times'].shape[0]).type(torch.long)
                St = None
                St_var = None
                prev_time = 0
                model.reset_context()
                U_starts = trj['U'][:, 0]
                U_ends = trj['U'][:, 1]
                U_vals = trj['U'][:, 2]
                SI_times = trj['SI'][1:, 0]
                SI_vals = trj['SI'][1:, 1:]
                model.set_context(trj['SI'][0, 1:].to(model.device))
                for id in (traj_id):
                    ttimes = trj['times'][id].to(model.device)
                    if isinstance(prev_time,torch.Tensor):
                        prev_time = prev_time.item()
                    U_idx = U_starts <= trj['times'][id]
                    U_idx = torch.logical_and(U_idx, U_ends > prev_time)
                    U_start = torch.clip(U_starts[U_idx],min=torch.Tensor([prev_time]).item())
                    U_end = torch.clip(U_ends[U_idx],max=ttimes.item())
                    U_val = U_vals[U_idx].to(model.device)
                    SI_idx = SI_times >= prev_time
                    SI_idx = torch.logical_and(SI_idx,SI_times < trj['times'][id])
                    SI_time = SI_times[SI_idx]
                    SI_val = SI_vals[SI_idx]
                    SI_idx = torch.ones_like(SI_time).type(torch.bool)
                    for act in range(torch.sum(U_idx)):
                        if U_start[act] > prev_time:
                            valid_SI_idx = SI_time[SI_idx] < U_start[act]
                            if torch.any(valid_SI_idx):
                                for jj, si in enumerate(SI_val[SI_idx][valid_SI_idx]):
                                    model.set_context(si.to(model.device))
                            St, St_var = model(St, St_var, (U_start[act].to(model.device) - prev_time).view(-1, 1),
                                               U=torch.zeros_like(U_val[act].view(1, -1)))
                            SI_idx = torch.logical_and(SI_idx,SI_time >= U_start[act])
                            prev_time = U_start[act]
                        valid_SI_idx = SI_time[SI_idx] < U_end[act]
                        if torch.any(valid_SI_idx):
                            for jj, si in enumerate(SI_val[SI_idx][valid_SI_idx]):
                                model.set_context(si.to(model.device))
                        St, St_var = model(St, St_var, (U_end[act].to(model.device) - prev_time).view(-1,1), U=U_val[act].view(1,-1))
                        SI_idx = torch.logical_and(SI_idx, SI_time >= U_end[act])
                        prev_time = U_end[act]
                    if prev_time < trj['times'][id]:
                        valid_SI_idx = SI_time[SI_idx] < trj['times'][id]
                        if torch.any(valid_SI_idx):
                            for jj, si in enumerate(SI_val[SI_idx][valid_SI_idx]):
                                model.set_context(si.to(model.device))
                        St, St_var = model(St, St_var, (ttimes - prev_time).view(-1, 1),
                                           U=torch.zeros_like(trj['U'][0,2].view(1, -1)).to(model.device))
                    if St is None:
                        St, St_var = model.get_prior()
                    prev_time = ttimes

                    obs_mask = trj['mask'][id].type(torch.bool)
                    dim_obs = torch.sum(obs_mask)
                    smp_mask = torch.zeros(2 * model.m + model.n).type(torch.bool)
                    smp_mask[:model.m] = obs_mask
                    smp = trj['obs'][id][obs_mask].view(-1,dim_obs).to(model.device)

                    if deterministic:
                        pred_mu = St.view(-1, dim_obs)
                        mseloss = model.loss_fn(pred_mu, smp)
                        loss = torch.zeros_like(mseloss)
                    else:
                        pred_mu = St[..., smp_mask].view(-1, dim_obs)
                        pred_var = torch.abs(torch.diagonal(St_var, dim1=-2, dim2=-1)[..., smp_mask].view(-1, dim_obs))
                        loss = model.loss_fn(pred_mu, smp, pred_var)
                        mseloss = mse_fn(pred_mu.view(-1),smp.view(-1))
                    if torch.isnan(loss):
                        loss = torch.zeros(1,device=model.device,requires_grad=True)
                    if not deterministic:
                        St, St_var = model.conditional_dist(St, St_var, smp_mask, smp)

                    if NLL is None:
                        NLL = (1.0 / len(traj_id)) * loss
                        MSE = (1.0 / len(traj_id)) * mseloss
                    else:
                        NLL = NLL + (1.0 / len(traj_id)) * loss
                        MSE = MSE + (1.0 / len(traj_id)) * mseloss
            NLL = (1.0 / b_size) * NLL
            MSE = (1.0 / b_size) * MSE
            mean_loss = mean_loss + NLL.detach()
            mean_msl = mean_msl + MSE.detach()
            if deterministic:
                MSE.backward()
            else:
                NLL.backward()
            optim.step()

            epochs.append(t)
            iterations.append(iter)
            groups.append('train')
            NLLs.append(NLL.item())
            MSEs.append(MSE.item())
        mean_loss = (1.0 / n_batches) * mean_loss
        mean_msl = (1.0 / n_batches) * mean_msl
        if verbose >= 1:
            print("Train loss at epoch " + str(t + 1) + ": NLL: "  + str(mean_loss.item()) + ", MSE: " + str(mean_msl.item()))
        if (t + 1) % eval_rate == 0:
            dl.valid()
            valid_nll, valid_mse, _ = eval_contextual_NESDE(model, dl, deterministic=deterministic, skip_first=skip_first)
            valid_nll, valid_mse = np.mean(valid_nll), np.mean(valid_mse)

            epochs.append(t)
            iterations.append(iter)
            groups.append('valid')
            NLLs.append(valid_nll)
            MSEs.append(valid_mse)
            if verbose >= 1:
                print("Validation loss at epoch " + str(t + 1) + ": NLL: " + str(valid_nll) + ", MSE: " + str(valid_mse) + f" [{time.time()-t0:.0f}s]")

            if valid_nll < best_valid_loss:
                best_valid_loss = valid_nll
                pat_count = 0
                torch.save(model.state_dict(), w_path)
            else:
                pat_count += 1

            if pat_count > patience:
                print("Met stopping criteria at epoch " + str(t + 1) + ". Stopping!")
                model.load_state_dict(torch.load(w_path))
                return train_res_summary(epochs, iterations, groups, NLLs, MSEs)
        if sched is not None:
            sched.step()

    model.load_state_dict(torch.load(w_path))
    return train_res_summary(epochs, iterations, groups, NLLs, MSEs)


def eval_LSTM(model, dl, eval_on_masked=True, verbose=0, detailed_res=False, Tmin=None):
    with torch.no_grad():
        t0 = time.time()
        model.eval()
        VARs, MSEs, CORRs = [], [], []
        tTRAJ, tTIMEs, tY, tP, tMSEs = [], [], [], [], []
        n_batches = 0
        epoch_done = False
        while not epoch_done:
            batch, epoch_done, traj_ids = dl.get_batch(return_ids=True)
            n_batches += 1
            b_size = 0
            for trj, j in zip(batch, traj_ids):
                b_size += 1
                model.init()
                preds = model(trj, Tmin).cpu().numpy()
                y = trj['obs'].cpu().numpy()[1:]
                times = trj['times'].cpu().numpy()[1:]
                mask = trj['mask'].type(torch.bool).cpu().numpy()[1:]
                if eval_on_masked:
                    times, preds, y = times[mask], preds[mask], y[mask]
                if Tmin is not None:
                    time_mask = times > Tmin
                    times, preds, y = times[time_mask], preds[time_mask], y[time_mask]
                dy = np.diff(y)
                se = (preds - y)**2
                trj_mse = np.mean(se)

                VARs.append(np.mean(dy**2))
                MSEs.append(trj_mse)
                CORRs.append(np.corrcoef(preds.reshape(-1), y.reshape(-1))[0,1])
                if detailed_res:
                    tTRAJ.extend(len(times)*[j])
                    tTIMEs.extend(list(times))
                    tY.extend(list(y.reshape(-1)))
                    tP.extend(list(preds.reshape(-1)))
                    tMSEs.extend(list(se.reshape(-1)))

        if verbose >= 1:
            print(f'Test:\tMSE={np.mean(MSEs):.4f}\t({time.time()-t0:.0f}s)')

        if detailed_res:
            return VARs, MSEs, CORRs, tTIMEs, tTRAJ, tMSEs
        return VARs, MSEs, CORRs

def train_LSTM(model, dl, optim, train_epochs, w_path, eval_rate=1, patience=200, eval_on_masked=True,
               sched=None, verbose=1, log_rate=10):
    iterations, epochs, groups, MSEs = [], [], [], []
    iter = -1
    pat_count = 0
    best_valid_loss = np.inf
    t0 = time.time()
    for t in range(train_epochs):
        dl.train()
        model.train()
        n_batches = 0
        epoch_mse = 0
        epoch_done = False
        while not epoch_done:  # for batch in epoch
            iter += 1
            batch, epoch_done = dl.get_batch()
            n_batches += 1
            optim.zero_grad()
            b_size = 0
            batch_mse = None
            for trj in batch:
                b_size += 1
                model.init()
                preds = model(trj)
                y = trj['obs'].to(model.device)[1:]
                mask = trj['mask'].to(model.device).type(torch.bool)[1:]
                if eval_on_masked:
                    preds, y = preds[mask], y[mask]
                se = torch.square(preds - y)
                trj_mse = torch.mean(se)
                batch_mse = trj_mse if batch_mse is None else batch_mse + trj_mse
            batch_mse = batch_mse / b_size
            batch_mse.backward()
            optim.step()

            epoch_mse += batch_mse.item()
            epochs.append(t)
            iterations.append(iter)
            groups.append('train')
            MSEs.append(batch_mse.item())

        epoch_mse = epoch_mse / n_batches
        if verbose >= 2:
            print("Train loss at epoch " + str(t + 1) + ": " + str(epoch_mse))

        if (t + 1) % eval_rate == 0:
            dl.valid()
            valid_var, valid_mse, _ = eval_LSTM(model, dl)
            valid_var, valid_mse = np.mean(valid_var), np.mean(valid_mse)

            epochs.append(t)
            iterations.append(iter)
            groups.append('valid')
            MSEs.append(valid_mse)
            if verbose >= 1 and (t%(eval_rate*log_rate)) == 0:
                print(f"[validation] epoch {t:02d}:\tMSE={valid_mse:.4f}\tEV={100*(1-valid_mse/valid_var):.2f}%\t({time.time()-t0:.0f}s)")

            valid_loss = valid_mse
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                pat_count = 0
                torch.save(model.state_dict(), w_path)
            else:
                pat_count += 1

            if pat_count > patience:
                print("Met stopping criteria after " + str(t + 1) + " epochs. Stopping!")
                model.load_state_dict(torch.load(w_path))
                return train_res_summary(epochs, iterations, groups, len(MSEs)*[None], MSEs)
        if sched is not None:
            sched.step()

    model.load_state_dict(torch.load(w_path))
    return train_res_summary(epochs, iterations, groups, len(MSEs)*[None], MSEs)


def train_res_summary(epochs, iterations, groups, NLLs, MSEs):
    return pd.DataFrame(dict(
        epoch = epochs,
        iteration = iterations,
        group = groups,
        NLL = NLLs,
        MSE = MSEs
    ))
