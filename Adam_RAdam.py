import math
import torch
from torch.optim.optimizer import Optimizer, required


# AdamW that have mode 'RAdam'
class Adam_RAdam(Optimizer):
    def __init__(self, params, mode="RAdam", lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, do_buffer=True, var_rate=1.0):
        """
        Argumetns:
        - params: Parameters to optimize
        - mode: Parameter for optimizer mode. Mode should be 'RAdam' or 'AdamW'. Default: 'RAdam', Type: string
        - lr: Learning rate on Neural Network. Default: 1e-3, Type: float
        - betas: Decay rate to calculate moving average and moving 2nd moment. Default: (0.9, 0.999), Type: Tuple(float, float)
        - eps: Epsilon. Used to calibrate too small values. Default: 1e-8, Type: float
        - weight_decay: Weight decay rate. Default: 0, Type: float
        - do_buffer: Whether to cache or not. Default: True, Type: bool
        - var_rate: Parameters for when to change optimizer mode from RAdam to Adam. Default: 1.0, Type: float
        """

        # Chaching the parameters conditions.
        if 0.0 > lr:
            raise ValueError(f"Learning rate (lr) must be a number greater than or equal to zero. {lr} is wrong input.")
        if betas[0] >= 1.0 or betas[0] < 0.0:
            raise ValueError(f"Beta1 must be a number between 0 and less than 1. {betas[0]} is wrong input.")
        if betas[1] >= 1.0 or betas[1] < 0.0:
            raise ValueError(f"Beta2 must be a number between 0 and less than 1. {betas[1]} is wrong input.")
        if eps <= 0.0:
            raise ValueError(f"Epsilon (eps) must be a number greater than zero. {eps} is wrong input.")
        if weight_decay >= 1.0 or weight_decay < 0.0:
            raise ValueError(
                f"Weight decay rate (weight_decay) must be a number between 0 and less than 1. {weight_decay} is wrong "
                f"input.")
        if not (mode == "RAdam" or mode == "AdamW"):
            raise ValueError(f"Mode (mode) should be 'RAdam' or 'AdamW'. {mode} is wrong input.")
        if type(do_buffer) is not bool:
            raise TypeError(f"do_buffer must be a bool type. {type(do_buffer)} is wrong type.")
        if var_rate < 0.0 or var_rate > 1.0:
            raise ValueError(f"Variation rate must be a number between 0 and 1, inclusive. {var_rate} is wrong input.")

        # buffer is an array used for caching. [step, N_sma(Length of the approximated SMA), step_size]
        self.buffer = [None, None, None]

        defaults = dict(lr=lr, mode=mode, betas=betas, eps=eps, weight_decay=weight_decay,
                        do_buffer=do_buffer, var_rate=var_rate)
        super(Adam_RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_RAdam, self).__setstate__(state)

    # Change optimizer mode to AdamW
    def mode_to_AdamW(self):
        for group in self.param_groups:
            group['mode'] = "AdamW"

    # Change optimizer mode to RAdam
    def mode_to_RAdam(self):
        for group in self.param_groups:
            group['mode'] = "RAdam"

    def step(self, closure=None):
        """
        Argument:
        - closure: Callable method that reevaluate model and return loss.

        Variables:
        - grad: Gradient values (float)
        - p_data_fp32: Parameter's data (float)
        - exp_avg: Exponential moving 1st moment (float)
        - exp_avg_sq: Exponential moving 2nd moment (float)
        - N_sma: Length of the approximated SMA (float)
        - N_sma_max: Maximum length of the approximated SMA (float)
        - beta1_t = beta1 ^ t  t is a step (float)
        - beta2_t = beta2 ^ t  t is a step (float)
        - bias_correction1: bias_corrected moving average  (float)
        - bias_correction2: bias_corrected moving 2nd moment  (float)
        - rect: Variance rectification term
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data.float()
                p_data_fp32 = p.data.float()
                state = self.state[p]
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                # initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32).type_as(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32).type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # v_t = beta2 * v_(t-1) + (1 - beta2) * g_t * g_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # m_t = beta1 * m_(t-1) + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1

                buffered = self.buffer

                # If cache data is already stored, there is no need to recalculate except for the denominator
                # If 'do_buffer' is false, buffer will have [None, None, None]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]

                    if group['mode'] == "AdamW":
                        # Denominator is sqrt(v_t) + epsilon
                        denom = exp_avg_sq.sqrt().add(group['eps'])

                    elif N_sma >= 5:
                        # Denominator is sqrt(v_t) + epsilon
                        denom = exp_avg_sq.sqrt().add(group['eps'])

                    else:
                        # In this case, we don't need sqrt(v_t) in the calculation, so denominator is [1.0, ...]
                        denom = torch.ones_like(exp_avg_sq)

                else:
                    beta1_t = beta1 ** state['step']
                    beta2_t = beta2 ** state['step']
                    bias_correction1 = 1 - beta1_t
                    bias_correction2 = 1 - beta2_t

                    if group['mode'] == "RAdam":
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                        if N_sma >= 5:
                            rect = math.sqrt(((N_sma - 4) * (N_sma - 2) * N_sma_max) / ((N_sma_max - 4) * (N_sma_max - 2) * N_sma))

                            # If (r_t)^2 > var_rate then change optimizer mode to AdamW.
                            if (rect * rect) > group['var_rate']:
                                # AdamW's step size
                                step_size = math.sqrt(bias_correction2) / bias_correction1
                                self.mode_to_AdamW()
                            else:
                                step_size = rect * math.sqrt(bias_correction2) / bias_correction1

                            # Denominator is sqrt(v_t) + epsilon.
                            denom = exp_avg_sq.sqrt().add(group['eps'])
                        else:
                            step_size = 1.0 / bias_correction1

                            # In this case, we don't need sqrt(v_t) in the calculation, so denominator is [1.0, ...]
                            denom = torch.ones_like(exp_avg_sq)

                        # If caching is used, the value is temporarily saved.
                        if group['do_buffer']:
                            buffered[1] = N_sma
                    else:
                        step_size = math.sqrt(bias_correction2) / bias_correction1

                        # Denominator is sqrt(v_t) + epsilon.
                        denom = exp_avg_sq.sqrt().add(group['eps'])

                # If caching is used, the value is temporarily saved.
                if group['do_buffer']:
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss

