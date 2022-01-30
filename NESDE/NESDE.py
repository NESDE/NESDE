import torch
import torch.nn as nn
from .hypernet import ConfNet, SeqHyper, HyperNet

min_val_param = 1e-10

class ControlCall():
    def __init__(self, U_ar, op_U_t):
        self.U = U_ar
        self.Oper = op_U_t
    def __call__(self, t):
        return self.Oper(self.U,t)

def Phi_(times, lambdas, V, n, time_par=False):
    # computes the fundemental matrix for real eigenvalues/vectors
    if time_par:
        et = torch.diag_embed(torch.exp(times.view(-1, 1).repeat((1, n)) * lambdas))
        Phi = V @ torch.nan_to_num(et).view(-1, n, n)
        et_inv = torch.diag_embed(torch.exp(times.view(-1, 1).repeat((1, n)) * (-1) * lambdas))
        Phi_inv = torch.nan_to_num(et_inv).view(-1, n, n) @ torch.nan_to_num(torch.linalg.inv(V))
    else:
        if not isinstance(times,list):
            times = [times]
        Phi = []
        Phi_inv = []
        for t in times:
            et = torch.diag_embed(torch.exp(t.view(-1,1).repeat((1, n)) * lambdas))
            Phi.append(torch.bmm(V, torch.nan_to_num(et).view(-1, n, n)))
            et_inv = torch.diag_embed(torch.exp(t.view(-1, 1).repeat((1, n)) * (-1) * lambdas))
            Phi_inv.append(torch.bmm(torch.nan_to_num(et_inv).view(-1,n,n), torch.nan_to_num(torch.linalg.inv(V))))
    return Phi, Phi_inv

def Phi_comp_(times, lambdas, V, n, time_par=False):
    # computes the fundemental matrix for complex eigenvalues/vectors
    if time_par:
        et = torch.diag_embed(torch.exp(times.view(-1, 1).repeat((1, n // 2)).type(torch.cfloat) * lambdas))
        phi_comp = V @ et.view(-1, n // 2, n // 2)
        Phi = torch.cat([phi_comp.real, phi_comp.imag], dim=2)
        Phi_inv = torch.nan_to_num(torch.linalg.inv(Phi))
        Phi = torch.nan_to_num(Phi)
    else:
        if not isinstance(times,list):
            times = [times]
        Phi = []
        Phi_inv = []
        for t in times:
            et = torch.diag_embed(torch.exp(t.view(-1,1).repeat((1, n // 2)).type(torch.cfloat) * lambdas))
            phi_comp = torch.bmm(V, et.view(-1, n // 2, n // 2))
            Phi.append(torch.cat([phi_comp.real,phi_comp.imag],dim=2))
            Phi_inv.append(torch.nan_to_num(torch.linalg.inv(Phi[-1])))
            Phi[-1] = torch.nan_to_num(Phi[-1])
    return Phi, Phi_inv

def integ_exp_trig(lambdas,thetas,t,t0=None):
    if t0 is None:
        t0 = torch.zeros_like(t)
    exps_t = torch.nan_to_num(torch.exp(lambdas * t))
    exps_t0 = torch.nan_to_num(torch.exp(lambdas * t0))
    denominator = torch.square(lambdas) + torch.square(thetas)
    cos_t = torch.cos(thetas * t)
    cos_t0 = torch.cos(thetas * t0)
    sin_t = torch.sin(thetas * t)
    sin_t0 = torch.sin(thetas * t0)

    res_cos_t = exps_t * (lambdas * cos_t + thetas * sin_t)
    res_cos_t0 = exps_t0 * (lambdas * cos_t0 + thetas * sin_t0)
    res_cos = (1.0 / torch.clip(denominator,min=min_val_param)) * torch.sign(denominator) * (res_cos_t - res_cos_t0)

    res_sin_t = exps_t * (lambdas * sin_t - thetas * cos_t)
    res_sin_t0 = exps_t0 * (lambdas * sin_t0 - thetas * cos_t0)
    res_sin = (1.0 / torch.clip(denominator,min=min_val_param)) * torch.sign(denominator) * (res_sin_t - res_sin_t0)
    return torch.nan_to_num(res_cos), torch.nan_to_num(res_sin) 

var_ind11 = torch.Tensor([[1,0],[0,0]]).type(torch.bool)
var_ind12 = torch.Tensor([[0,1],[0,0]]).type(torch.bool)
var_ind21 = torch.Tensor([[0,0],[1,0]]).type(torch.bool)
var_ind22 = torch.Tensor([[0,0],[0,1]]).type(torch.bool)
def analytic_int(exps,t,complex,var,t0=None, exps_conj=None,ext_prod=None):
    # analytical integration
    if complex:
        if var:
            expon = exps.real
            trig1 = exps.imag
            trig2 = exps_conj.imag
            cos_1, sin_1 = integ_exp_trig((-1.0) * expon, trig1,t,t0)
            cos_2, sin_2 = integ_exp_trig((-1.0) * expon, trig2,t,t0)
            var_cos_square = 0.5 * (cos_1 + cos_2)
            var_sin_cos = 0.5 * (sin_1 + sin_2)
            var_cos_sin = 0.5 * (sin_1 - sin_2)
            var_sin_square = 0.5 * (cos_2 - cos_1)
            n = ext_prod.shape[-1] // 2
            ind11 = var_ind11.repeat(n,n)
            ind12 = var_ind12.repeat(n,n)
            ind21 = var_ind21.repeat(n,n)
            ind22 = var_ind22.repeat(n,n)
            ind_cor = torch.ones(n,n).type(torch.bool)
            dot_11 = ext_prod[...,ind11].view(-1,n,n)
            dot_12 = ext_prod[...,ind12].view(-1,n,n)
            dot_21 = ext_prod[...,ind21].view(-1,n,n)
            dot_22 = ext_prod[...,ind22].view(-1,n,n)
            res_11 = (var_cos_square * dot_11) - (var_sin_cos * dot_21) - (var_cos_sin * dot_12) + (var_sin_square * dot_22)
            res_12 = (var_cos_sin * dot_11) - (var_sin_square * dot_21) + (var_cos_square * dot_12) - (var_sin_cos * dot_22)
            res_21 = (var_sin_cos * dot_11) + (var_cos_square * dot_21) - (var_sin_square * dot_12) - (var_cos_sin * dot_22)
            res_22 = (var_sin_square * dot_11) + (var_cos_sin * dot_21) + (var_sin_cos * dot_12) + (var_cos_square * dot_22)
            res = torch.zeros_like(ext_prod)
            res[...,ind11] = res_11[...,ind_cor]
            res[...,ind12] = res_12[...,ind_cor]
            res[...,ind21] = res_21[...,ind_cor]
            res[...,ind22] = res_22[...,ind_cor]
            return res
        else:
            expon = exps.real
            trig = exps.imag
            n = ext_prod.shape[-2] // 2
            ind11 = torch.block_diag(*[var_ind11 for _ in range(n)])
            ind12 = torch.block_diag(*[var_ind12 for _ in range(n)])
            ind21 = torch.block_diag(*[var_ind21 for _ in range(n)])
            ind22 = torch.block_diag(*[var_ind22 for _ in range(n)])
            ind_cor = torch.ones(n).type(torch.bool)
            cos_int, sin_int = integ_exp_trig((-1.0) * expon, trig,t,t0)
            mul_int = torch.zeros_like(ext_prod).view(-1,2*n,1).repeat(1,1,2*n)
            mul_int[...,ind11] = cos_int[...,ind_cor]
            mul_int[...,ind12] = (-1.0) * sin_int[...,ind_cor]
            mul_int[...,ind21] = sin_int[...,ind_cor]
            mul_int[...,ind22] = cos_int[...,ind_cor]
            res = torch.bmm(mul_int, ext_prod)
            return res
    else:
        if t0 is None:
            integ = (1.0 / torch.clip(torch.abs(exps),min=min_val_param)) * torch.sign(exps) * (torch.exp(exps * t) - 1.0)
        else:
            integ = (1.0 / torch.clip(torch.abs(exps),min=min_val_param)) * torch.sign(exps) * (torch.exp(exps * (t - t0)) - 1.0)
        return torch.nan_to_num(integ)


class ESDE():
    def __init__(self, n, device, complex=False):
        # n - system's dimension
        # device - device to run on
        # complex - weather or not use complex eigenvalues and eigenvectors
        #####################################################################################################
        # choose the right form of fundemental matrix:
        if complex:
            assert n % 2 == 0
            self.Phi = Phi_comp_
        else:
            self.Phi = Phi_
        self.n = n
        self.complex = complex
        self.device = device

    def predict_an(self, X0, X0_var, U, times, params):
        # ESDE - analytical version. good for a piecewise-constant control (or no control)
        # X0 - initial state at t=0
        # X0_var - covariance matrix at t=0
        # U an action that is constant within the time interval, of dimension n, shape (B,n), where B is batch size
        # times - time to calculate, shape (B,1)
        # params - a dict that contains V and lambda, the first dimension corresponds to the batch dim
        #####################################################################################################
        # float64 for nummerical stability:
        X0 = X0.type(torch.float64)
        X0_var = X0_var.type(torch.float64)
        U = U.type(torch.float64)
        times = times.type(torch.float64)
        Q = params['Q'].type(torch.float64)

        X0 = X0.view(-1,self.n,1)
        if self.complex:
            lambdas = params['lambda'].type(torch.complex128)
            V = params['V'].type(torch.complex128)

            # compute initial conditions:
            Phi0, Phi0_inv = self.Phi(torch.zeros_like(times), lambdas, V, self.n)
            Phi0_inv = Phi0_inv[0]
            init_cond = torch.bmm(Phi0_inv, X0)
            init_var = torch.bmm(torch.bmm(Phi0_inv, X0_var), Phi0_inv.transpose(-2, -1))

            # compute the fundemental matrix at the given time:
            Phi, _ = self.Phi(times, lambdas, V, self.n)
            Phi = Phi[0].view(-1, self.n, self.n)

            # Homogeneous solution:
            Hom = torch.bmm(Phi, init_cond)
            Hom_var = torch.bmm(torch.bmm(Phi, init_var), Phi.transpose(-2, -1))
            Hom_var = Hom_var * (
                    torch.diag_embed(torch.diagonal(torch.sign(Hom_var), dim1=-2, dim2=-1)) + torch.ones_like(Hom_var)
                    - torch.eye(self.n, device=self.device).view(1, self.n, self.n).repeat(Hom_var.shape[0], 1, 1))
            ext_prod_lambdas = torch.bmm(Phi0_inv, (U.view(-1,self.n,1)))
            lambda_int = analytic_int(lambdas,times.view(-1,1),self.complex,False,ext_prod=ext_prod_lambdas).view(-1,self.n,1)
            Part = torch.bmm(Phi,lambda_int)
            ext_prod_siggs = torch.bmm(Phi0_inv, torch.bmm(Q, Phi0_inv.transpose(-1,-2)))
            lambdas_siggs = lambdas.unsqueeze(1).repeat(1,lambdas.shape[-1],1) + lambdas.unsqueeze(-1).repeat(1,1,lambdas.shape[-1])
            lambdas_siggs_conj = lambdas.unsqueeze(1).repeat(1, lambdas.shape[-1], 1) - lambdas.unsqueeze(-1).repeat(1, 1, lambdas.shape[-1])
            sig_int = analytic_int(lambdas_siggs,times.view(-1,1,1),self.complex,True, exps_conj=lambdas_siggs_conj,ext_prod=ext_prod_siggs)
            Part_var = torch.bmm(Phi, torch.bmm(sig_int, Phi.transpose(-1, -2)))
        else:
            lambdas = params['lambda'].type(torch.float64)
            V = params['V'].type(torch.float64)

            # compute initial conditions:
            Phi0, Phi0_inv = self.Phi(torch.zeros_like(times), lambdas, V, self.n)
            Phi0_inv = Phi0_inv[0]
            Phi0 = Phi0[0]
            init_cond = torch.bmm(Phi0_inv, X0)
            init_var = torch.bmm(torch.bmm(Phi0_inv, X0_var), Phi0_inv.transpose(-2, -1))

            # compute the fundemental matrix at the given time:
            Phi, _ = self.Phi(times, lambdas, V, self.n)
            Phi = Phi[0].view(-1, self.n, self.n)

            # Homogeneous solution:
            Hom = torch.bmm(Phi, init_cond)
            Hom_var = torch.bmm(torch.bmm(Phi, init_var), Phi.transpose(-2, -1))
            Hom_var = Hom_var * (
                    torch.diag_embed(torch.diagonal(torch.sign(Hom_var), dim1=-2, dim2=-1)) + torch.ones_like(
                Hom_var) - torch.eye(self.n, device=self.device).view(1, self.n, self.n).repeat(Hom_var.shape[0], 1, 1))
            lambda_int = analytic_int(lambdas,times.view(-1,1),self.complex,False).view(-1,self.n,1)
            Part = torch.bmm(Phi0,lambda_int * torch.bmm(Phi0_inv, (U.view(-1,self.n,1))))
            lambdas_siggs = lambdas.unsqueeze(1).repeat(1,lambdas.shape[-1],1) + lambdas.unsqueeze(-1).repeat(1,1,lambdas.shape[-1])
            sig_int = analytic_int(lambdas_siggs,times.view(-1,1,1),self.complex,True)
            Part_var = sig_int * torch.bmm(Phi0_inv, torch.bmm(Q, Phi0_inv.transpose(-1,-2)))
            Part_var = torch.bmm(Phi0, torch.bmm(Part_var, Phi0.transpose(-1, -2)))
            Part_var = Part_var * (torch.diag_embed(torch.diagonal(torch.sign(Part_var),dim1=-2,dim2=-1)) + torch.ones_like(Part_var)
                                   - torch.eye(self.n,device=self.device).view(1,self.n,self.n).repeat(Part_var.shape[0],1,1))

        Sol = Hom.view(-1,self.n) + Part.view(-1, self.n)
        Sol_var = (Hom_var + Part_var).view(-1, self.n, self.n)
        return Sol.type(torch.get_default_dtype()), Sol_var.type(torch.get_default_dtype())

    def predict_num(self, X0, X0_var, U, times, params, dt, t0_ac=None):
        # ESDE - numerical version. good for an arbitrary control signal
        # X0 - initial state at t=0
        # X0_var - covariance matrix at t=0
        # U a callable that maps t to an action of dimension n, shape (B,n), where B is batch size
        # times - time to calculate, shape (B,1)
        # params - a dict that contains V and lambda, the first dimension corresponds to the batch dim
        # dt - a time resolution for the integration
        #####################################################################################################
        X0 = X0.view(-1, self.n, 1)
        # compute initial conditions:
        _, Phi0_inv = self.Phi(torch.zeros_like(times), params['lambda'], params['V'], self.n)
        Phi0_inv = Phi0_inv[0]
        init_cond = torch.bmm(Phi0_inv, X0)
        init_var = torch.bmm(torch.bmm(Phi0_inv, X0_var), Phi0_inv.transpose(-2, -1))
        # compute the fundemental matrix and its inverse for all times
        n_steps = (times / dt).type(torch.long) + 1
        ttimes = [dt * torch.arange(n_steps[j].item(), device=self.device).view(-1, 1) for j in range(params['lambda'].shape[0])]
        Sol = []
        Sol_var = []
        if t0_ac is None:
            t0_ac = torch.zeros_like(times).view(-1)
        for b, btimes in enumerate(ttimes):
            U_num = U[b](t0_ac[b] + torch.cat([torch.arange(n_steps[b].item(),device=self.device)*dt, times[b].view(1)],dim=0)).view(-1, self.n, 1)
            Phi, Phi_inv = self.Phi(torch.cat([btimes.view(-1), times[b].view(1)],dim=0), params['lambda'][b], params['V'][b], self.n, time_par=True)
            Phi_t = Phi[-1]
            # Homogeneous solution:
            Hom = Phi_t @ init_cond[b]
            Hom_var = Phi_t @ init_var[b] @ Phi_t.transpose(-2, -1)

            # integrands to sum:
            integrand = torch.bmm(Phi_inv, U_num).view(-1, self.n)
            integrand_var = torch.bmm(Phi_inv @ params['Q'][b], Phi_inv.transpose(-2, -1)).view(-1, self.n, self.n)

            # integrate to obtain non-homogeneous solution ("Part"):
            frac_dt = (times[b] - dt *(n_steps[b]-1)).view(1)
            integrand[-1] = (frac_dt / dt) * integrand[-1]
            integ = dt * torch.sum(integrand,dim=0)
            integrand_var[-1] = (frac_dt / dt) * integrand_var[-1]
            integ_var = dt * torch.sum(integrand_var,dim=0)
            Part = Phi_t @ integ
            Part_var = Phi_t @ integ_var @ Phi_t.transpose(-2, -1)

            # solution is sum of Hom and Part:
            Sol.append(Hom.view(1, self.n) + Part.view(1, self.n))
            Sol_var.append((Hom_var + Part_var).view(1, self.n, self.n))

        return torch.cat(Sol,dim=0), torch.cat(Sol_var,dim=0)

class NESDE(nn.Module):
    def __init__(self, n, m, params_interval, device, control_dim=0, lambda_hidden=[16,16], V_hidden=[16,16], Q_hidden=[16,16],
                 complex=False, dt=None, nonlinearity=nn.ReLU, stationary=False, stable=True, B=None, lambdas_fac=None, dropout=0.1,
                 bias_fac=1.0,var_fac=1.0, Q_fac=1.0):
        super(NESDE,self).__init__()
        # n - system's dim
        # m - observable dim
        # params_interval - delta T for parameters update
        # *_hidden - hidden layers for each module
        # device - using for tensor init
        # control_dim - dimension of control signal, 0 for no control
        # complex - complex eigenvalues and eigenvectors
        # dt - nummerical integration resolution, None for analytical integration.
        # nonlinearity - nonlinearity to be used within the hidden layers
        # stable - keep the eigenvalues negative to obtain stability
        # B - optional, a linear transformation for the control, should be of shape (control_dim,n)
        super(NESDE, self).__init__()
        self.params_interval = params_interval
        self.n = n
        self.m = m
        self.complex = complex
        self.control_dim = control_dim
        self.esde = ESDE(n, device, complex)
        self.dt = dt
        self.device = device
        self.stationary = stationary
        self.lambdas_fac = lambdas_fac
        self.bias_fac = bias_fac
        self.var_fac = var_fac
        self.Q_fac = Q_fac
        self.curr_params = None

        lambda_layers = []
        V_layers = []
        Q_layers = []
        lambda_sizes = [n] + lambda_hidden
        V_sizes = [n] + V_hidden
        Q_sizes = [n] + Q_hidden

        for i in range(1, len(lambda_sizes)):
            lambda_layers.extend([nn.Linear(lambda_sizes[i - 1], lambda_sizes[i], bias=True), nonlinearity()])
        if stable:
            lambda_layers.extend([nn.Linear(lambda_sizes[-1], n), nn.Sigmoid()])
        else:
            lambda_layers.extend([nn.Linear(lambda_sizes[-1], n), nn.Tanh()])

        for i in range(1, len(V_sizes)):
            V_layers.extend([nn.Linear(V_sizes[i - 1], V_sizes[i], bias=True), nonlinearity()])
        V_layers.extend([nn.Linear(V_sizes[-1], n ** 2),nn.Tanh(),nn.Linear(n**2,n**2)])

        for i in range(1, len(Q_sizes)):
            Q_layers.extend([nn.Linear(Q_sizes[i - 1], Q_sizes[i], bias=True), nonlinearity()])
        Q_layers.extend([nn.Linear(Q_sizes[-1], n**2),nn.Tanh()])


        self.dropout = nn.Dropout(p=dropout)
        self.lambdas = nn.Sequential(*lambda_layers)
        self.V = nn.Sequential(*V_layers)
        self.Q = nn.Sequential(*Q_layers)
        self.B = None
        if control_dim > 0:
            if B is None:
                self.B = nn.Parameter(torch.randn(control_dim,n))
            else:
                self.B = B.to(device)

        self.prior_mu = nn.Parameter(torch.randn(n+m))
        self.prior_var = nn.Parameter(torch.randn(n+m))
        self.loss_fn = nn.GaussianNLLLoss(full=True)
        self.to(device)
        if lambdas_fac is None:
            if complex:
                self.lambdas_fac = 2.0
            else:
                self.lambdas_fac = 1.0

    def pack_params(self,Xt, params=None, mask=None):
        # calculates the neural parameters for the ESDE
        # returns a dict of them
        net_input = Xt
        if self.stationary:
            net_input = torch.ones_like(net_input)
        net_input = self.dropout(torch.nan_to_num(net_input))
        if self.complex:
            V = self.V(net_input).view(-1,self.n,self.n // 2,2)
            V = V / torch.clip(torch.linalg.norm(V,dim=1).unsqueeze(1).repeat(1,self.n,1,1),min=min_val_param)
            V = torch.view_as_complex(V.view(-1,self.n,self.n // 2,2))
            lambdas = self.lambdas(net_input).view(-1,self.n // 2,2)

            lambdas = (-1.0) * self.lambdas_fac * torch.conj(torch.view_as_complex(lambdas))
            Q = torch.tril(self.Q(net_input).view(-1, self.n, self.n))
            Q = self.Q_fac * torch.bmm(Q, Q.transpose(-2, -1))
        else:
            V = self.V(net_input).view(-1,self.n,self.n)
            V = V / torch.clip(torch.linalg.norm(V,dim=1).unsqueeze(1).repeat(1,self.n,1),min=min_val_param)
            lambdas = (-1.0) * (self.lambdas_fac * self.lambdas(net_input))
            Q = torch.tril(self.Q(net_input).view(-1,self.n,self.n))
            Q = self.Q_fac * torch.bmm(Q,Q.transpose(-2,-1))
        if params is not None:
            not_mask = torch.logical_not(mask).type(torch.float)
            mask = mask.type(torch.float)
            V = V * mask.view(-1,1,1).repeat(1,V.shape[1],V.shape[2]) + params['V'] * not_mask.view(-1,1,1).repeat(1,V.shape[1],V.shape[2])
            lambdas = lambdas * mask.view(-1,1).repeat(1,lambdas.shape[1]) + params['lambda'] * not_mask.view(-1,1).repeat(1,lambdas.shape[1])
            Q = Q * mask.view(-1,1,1).repeat(1,Q.shape[1],Q.shape[2]) + params['Q'] * not_mask.view(-1,1,1).repeat(1,Q.shape[1],Q.shape[2])
        return {'V': V, 'lambda': lambdas, 'Q': Q}

    def esde_cal(self, X0, X0_var, times, params ,U, t0_ac):
        if self.dt is None:
            if self.B is None or U is None:
                BU = torch.zeros(times.shape[0], self.n, device=self.device)
            else:
                BU = U @ self.B
            Xt, Xt_var = self.esde.predict_an(X0, X0_var, BU, times, params)
        else:
            if self.B is None or U is None:
                BU = [lambda t: torch.zeros(times.shape[0], self.n, device=self.device) for b in range(X0.shape[0])]
            else:
                BU = [ControlCall(U[b].U.view(-1,self.B.shape[0]) @ self.B, U[b].Oper) for b in range(len(U))]
            Xt, Xt_var = self.esde.predict_num(X0, X0_var, BU, times, params,self.dt,t0_ac=t0_ac)
        return Xt, Xt_var

    def forward(self, S0, S0_var, times, U=None, t0_ac=(0,)):
        # S0 - initial state vector, of the form [X0+bias,bias,X0], when S0 is None the model uses prior
        # S0_var - initial covariance matrix, corresponding to S0 (can be None when S0 is None)
        # times - times to predict at
        # U - control signal (optional). should be constant when using analytical integration
        #      and callable (function of t) when using numerical integration
        # returns St, St_var (similar to S0, S0_var

        # use prior when no input is given:
        if S0 is None:
            Xt = self.prior_mu[:self.n].view(1,self.n).repeat(times.shape[0],1)
            bias = self.bias_fac * self.prior_mu[self.n:].view(1,self.m).repeat(times.shape[0],1)
            Xt_var = torch.diag_embed(torch.abs(self.prior_var[:self.n]).view(-1)).view(-1,self.n,self.n).repeat(times.shape[0],1,1)
            var = self.var_fac * torch.diag_embed(torch.abs(self.prior_var[self.n:]).view(-1)).view(-1,self.m,self.m).repeat(times.shape[0],1,1)
        else:
            Xt = S0[...,2*self.m:].view(-1, self.n).detach()
            Xt_var = S0_var[...,2*self.m:,2*self.m:].view(-1, self.n, self.n).detach()
            bias = S0[...,self.m:2*self.m:].view(-1, self.m)
            var = S0_var[...,self.m:2*self.m:,self.m:2*self.m:].view(-1, self.m, self.m)
        current_time = torch.zeros_like(times)
        params = None
        n_interval = ((1.0 / self.params_interval) * (times)).type(torch.long)
        for inter in range(1,torch.max(n_interval)+1):
            t_mask = n_interval >= inter
            not_t_mask = torch.logical_not(t_mask)
            params = self.pack_params(Xt,params=params,mask=t_mask)
            t_mask = t_mask.type(torch.float)
            not_t_mask = not_t_mask.type(torch.float)
            Xtt, Xtt_var = self.esde_cal(Xt, Xt_var, self.params_interval * torch.ones_like(times), params, U, [(inter - 1) * self.params_interval + t0_aci for t0_aci in t0_ac])

            Xt = Xtt * t_mask.view(-1,1).repeat(1,self.n) + Xt * not_t_mask.view(-1,1).repeat(1,self.n)
            Xt_var = Xtt_var * t_mask.view(-1,1,1).repeat(1,self.n,self.n) + Xt_var * not_t_mask.view(-1,1,1).repeat(1,self.n,self.n)
            current_time = current_time + self.params_interval * t_mask

        params = self.pack_params(Xt)
        self.curr_params = params
        Xt, Xt_var = self.esde_cal(Xt, Xt_var, times - current_time, params, U, [current_time[j] + t0_ac[j] for j in range(len(t0_ac))])
        Xt = Xt.view(-1,self.n)
        Xt_var = Xt_var.view(-1,self.n,self.n)
        St = torch.cat([Xt[:,:self.m] + bias,bias,Xt],dim=1)
        St_var = torch.cat([torch.cat([Xt_var[:,:self.m,:self.m] + var,var ,Xt_var[:,:self.m,:]],dim=2),
                            torch.cat([var,var,torch.zeros(var.shape[0],self.m,self.n,device=self.device)],dim=2),
                            torch.cat([Xt_var[:,:,:self.m], torch.zeros(var.shape[0],self.n,self.m,device=self.device),Xt_var],dim=2)],dim=1)
        return torch.nan_to_num(St), torch.nan_to_num(St_var)


    def conditional_dist(self, dist_mu, dist_sigma, smp_mask, smp):
        # calculates the conditional distribution of a multivariate gaussian
        # dist_mu - the mean of the variable (vector)
        # dist_sigma - the covariance (matrix)
        # smp_mask - boolean mask of the sample indices
        # smp - a given sample (of an arbitrary dimension, as lon as it matches the mask)
        not_smp_mask = torch.logical_not(smp_mask)
        mu1 = dist_mu[..., not_smp_mask]
        mu2 = dist_mu[..., smp_mask]
        sigma11 = dist_sigma[..., not_smp_mask, :][..., not_smp_mask]
        sigma12 = dist_sigma[..., not_smp_mask, :][..., smp_mask]
        sigma21 = dist_sigma[..., smp_mask, :][..., :, not_smp_mask]
        sigma22 = dist_sigma[..., smp_mask, :][..., :, smp_mask]

        mask = torch.linalg.det(sigma22) == 0
        mu_mask = mask.view(-1, 1).repeat(1, mu1.shape[1])
        sigma_mask = mask.view(-1, 1, 1).repeat(1, sigma11.shape[1], sigma11.shape[2])
        sigma22_mask = mask.view(-1, 1, 1).repeat(1, sigma22.shape[1], sigma22.shape[2])
        mu_cond = mu1 * mu_mask
        sigma_cond = sigma11 * sigma_mask
        sigma22_inv = torch.linalg.inv(
            sigma22_mask * torch.eye(sigma22.shape[-1], device=self.device).unsqueeze(0).repeat(sigma22.shape[0], 1, 1) + torch.logical_not(sigma22_mask) * sigma22)
        mu_cond = mu_cond + torch.logical_not(mu_mask) * (mu1 + torch.bmm(torch.bmm(sigma12, sigma22_inv), smp.view(1, -1, 1) - mu2.view(1, -1, 1)).squeeze(-1))
        sigma_cond = sigma_cond + torch.logical_not(sigma_mask) * (sigma11 - torch.bmm(torch.bmm(sigma12, sigma22_inv), sigma21))

        cond_dist_mu = torch.zeros_like(dist_mu)
        cond_dist_sigma = torch.zeros_like(dist_sigma)

        cond_dist_mu[...,smp_mask] = smp
        cond_dist_mu[...,not_smp_mask] = mu_cond

        not_id_sig = not_smp_mask.type(torch.float).view(-1,1) @ not_smp_mask.type(torch.float).view(1,-1)
        cond_dist_sigma[...,not_id_sig.type(torch.bool)] = sigma_cond.view(-1)
        cond_dist_sigma = cond_dist_sigma * (torch.diag_embed(torch.diagonal(torch.sign(cond_dist_sigma),dim1=-2,dim2=-1)) + torch.ones_like(cond_dist_sigma) - torch.eye(cond_dist_sigma.shape[-1],device=self.device).unsqueeze(0).repeat(cond_dist_sigma.shape[0],1,1))
        return cond_dist_mu, cond_dist_sigma



class ContEmbed(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, device, num_layers=3, dropout=0.1,lstm_dropout=0.1):
        super(ContEmbed, self).__init__()
        self.fc1 = nn.Sequential(*[nn.Dropout(p=dropout),nn.Linear(input_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=lstm_dropout)
        self.fc2 = nn.Sequential(*[nn.Linear(hidden_size, output_dim), nn.Tanh()])
        self.num_layers = num_layers
        self.lstm_size = hidden_size
        self.output_dim = output_dim
        self.device = device

    def forward(self, input, prev_state):
        if prev_state is None:
            prev_state = self.init_state(1)
        tmp = self.fc1(input)
        lstm_out, state = self.lstm(tmp, prev_state)
        out = self.fc2(torch.nan_to_num(lstm_out)[-1].view(1,self.lstm_size))
        return out, state

    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.lstm_size,device=self.device),
                torch.zeros(self.num_layers, batch_size, self.lstm_size,device=self.device))


class HyperNESDE(NESDE):
    def __init__(self, n, m, params_interval, device, context_dim, lstm_embed=False, embed_dim=None, embed_params=None,
                 hyper_hidden=[16,16],control_dim=0, lambda_hidden=[16,16], V_hidden=[16,16], Q_hidden=[16,16], complex=False,
                 dt=None, nonlinearity=nn.ReLU, stable=True, B=None, lambdas_fac=0.01, dropout=0.1, stationary=False, prior_hidden=[32,32],
                 bias_fac=100.0,var_fac=1000.0,Q_fac=0.4, features_data=None):
        # n - system's dim
        # m - observable dim
        # params_interval - delta T for parameters update
        # *_hidden - hidden layers for each module
        # device - using for tensor init
        # context_dim - dimension of embedded side information
        # lstm_embed - use internal lstm to embed the side information, if set to True, must set embed_dim
        # embed_dim - the raw side-information dmension
        # control_dim - dimension of control signal, 0 for no control
        # complex - complex eigenvalues and eigenvectors
        # dt - nummerical integration resolution, None for analytical integration.
        # nonlinearity - nonlinearity to be used within the hidden layers
        # stable - keep the eigenvalues negative to obtain stability
        # B - optional, a linear transformation for the control, should be of shape (control_dim,n)
        super(NESDE, self).__init__()
        self.params_interval = params_interval
        self.n = n
        self.m = m
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.complex = complex
        self.control_dim = control_dim
        self.esde = ESDE(n, device, complex)
        self.dt = dt
        self.device = device
        self.lstm_embed = lstm_embed
        self.stationary = stationary
        self.dropout = nn.Dropout(p=dropout)
        self.lambdas_fac = lambdas_fac
        self.bias_fac = bias_fac
        self.var_fac = var_fac
        self.Q_fac = Q_fac
        self.features_data = features_data
        lambda_layers = []
        V_layers = []
        Q_layers = []
        prior_layers = []
        lambda_sizes = [n] + lambda_hidden
        V_sizes = [n] + V_hidden
        Q_sizes = [n] + Q_hidden
        prior_sizes = [context_dim] + prior_hidden

        for i in range(1, len(lambda_sizes)):
            lambda_layers.extend([ConfNet(lambda_sizes[i - 1], lambda_sizes[i], bias=True), nonlinearity()])
        if stable:
            lambda_layers.extend([ConfNet(lambda_sizes[-1], n), nn.Sigmoid()])
        else:
            lambda_layers.extend([ConfNet(lambda_sizes[-1], n), nn.Tanh()])

        for i in range(1, len(V_sizes)):
            V_layers.extend([ConfNet(V_sizes[i - 1], V_sizes[i], bias=True), nonlinearity()])
        V_layers.extend([ConfNet(V_sizes[-1], n ** 2),nn.Tanh(),ConfNet(n**2,n**2)])

        for i in range(1, len(Q_sizes)):
            Q_layers.extend([ConfNet(Q_sizes[i - 1], Q_sizes[i], bias=True), nonlinearity()])
        Q_layers.extend([ConfNet(Q_sizes[-1], n**2),nn.Tanh()])

        for i in range(1, len(V_sizes)):
            prior_layers.extend([nn.Linear(prior_sizes[i - 1], prior_sizes[i], bias=True), nonlinearity()])
        prior_layers.extend([nn.Linear(prior_sizes[-1], 2*(n+m)),nn.Sigmoid()])

        if lstm_embed:
            self.lstm_state = None
            if embed_params is None:
                self.Embed = ContEmbed(embed_dim, context_dim, device=device, hidden_size=32, num_layers=1, dropout=0.2, lstm_dropout=0.0)
            else:
                self.Embed = ContEmbed(embed_dim,context_dim,device=device,hidden_size=embed_params['hidden_size'], num_layers=embed_params['num_layers'], dropout=embed_params['dropout'])

        self.lambdas = SeqHyper(lambda_layers)
        self.V = SeqHyper(V_layers)
        self.Q = SeqHyper(Q_layers)
        self.Hyper_lambdas = HyperNet(self.lambdas, context_dim, hyper_hidden, nonlinearity)
        self.Hyper_V = HyperNet(self.V, context_dim, hyper_hidden, nonlinearity)
        self.Hyper_Q = HyperNet(self.Q, context_dim, hyper_hidden, nonlinearity)
        self.prior_net = nn.Sequential(*prior_layers)
        self.B = None
        if control_dim > 0:
            if B is None:
                self.B = nn.Parameter(torch.randn(control_dim,n))
            else:
                self.B = B.to(device)

        self.loss_fn = nn.GaussianNLLLoss(full=True)
        self.to(device)
        if lambdas_fac is None:
            if complex:
                self.lambdas_fac = 2.0
            else:
                self.lambdas_fac = 1.0
    def forward(self, S0, S0_var, times, U=None, t0_ac=[0]):
        if S0 is None:
            self.rc_state = None
        St, St_var = super(HyperNESDE, self).forward(S0, S0_var, times, U, t0_ac)
        return St, St_var

    def conditional_dist(self, dist_mu, dist_sigma, smp_mask, smp):
        return super(HyperNESDE, self).conditional_dist(dist_mu, dist_sigma, smp_mask, smp)


    def set_context(self, input):
        if self.lstm_embed:
            if self.lstm_state is None:
                self.lstm_state = self.Embed.init_state(1)
            context, self.lstm_state = self.Embed(input.view(1,1,-1), self.lstm_state)
            context = context.view(-1)
        else:
            context = input.view(-1)
        self.Hyper_lambdas(context)
        self.Hyper_V(context)
        self.Hyper_Q(context)
        prior = self.prior_net(context).view(-1)
        self.prior_mu = prior[:self.n + self.m]
        self.prior_var = torch.abs(prior[self.n + self.m:])
        mask = torch.ones_like(self.prior_mu)
        mask[:self.n] = 0.0
        self.prior_mu = self.prior_mu * mask
        self.prior_var = self.prior_var * mask
        return

    def reset_context(self):
        if self.lstm_embed:
            self.lstm_state = None
        return

    def sample_context(self):
        feature = self.features_data[torch.randint(len(self.features_data),(1,))]
        feature_ids = torch.sort(torch.randperm(feature.shape[0])[:2]).values
        for i in torch.arange(feature_ids[0],feature_ids[1] + 1):
            self.set_context(feature[i].view(-1,self.embed_dim).to(self.device))
        weight = ((277.4 - 30.1) * feature[feature_ids[1], 4].cpu().numpy()) + 30.1
        return weight


    def get_prior(self):
        Xt = self.prior_mu[:self.n].view(1,self.n)
        bias = self.bias_fac * self.prior_mu[self.n:].view(1,self.m)
        Xt_var = torch.diag_embed(torch.abs(self.prior_var[:self.n]).view(-1)).view(-1,self.n,self.n)
        var = self.var_fac * torch.diag_embed(torch.abs(self.prior_var[self.n:]).view(-1)).view(-1,self.m,self.m)
        St = torch.cat([Xt[:,:self.m] + bias,bias,Xt],dim=1)
        St_var = torch.cat([torch.cat([Xt_var[:,:self.m,:self.m] + var,var ,Xt_var[:,:self.m,:]],dim=2),
                            torch.cat([var,var,torch.zeros(var.shape[0],self.m,self.n,device=self.device)],dim=2),
                            torch.cat([Xt_var[:,:,:self.m], torch.zeros(var.shape[0],self.n,self.m,device=self.device),Xt_var],dim=2)],dim=1)
        return St, St_var

