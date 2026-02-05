import torch
import torch.nn as nn
from .base import DQNModuleBase, DQN


class DQNModuleFeedforward(DQNModuleBase):

    def __init__(self, params):
        super(DQNModuleFeedforward, self).__init__(params)

        self.feedforward = nn.Sequential(
            nn.Linear(self.output_dim, params.hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x_screens, x_variables):
        """
        Argument sizes:
            - x_screens of shape (batch_size, seq_len * n_fm, h, w)
            - x_variables list of n_var tensors of shape (batch_size,)
        """
        batch_size = x_screens.size(0)
        assert x_screens.ndimension() == 4
        assert len(x_variables) == self.n_variables
        assert all(x.ndimension() == 1 and x.size(0) == batch_size for x in x_variables)

        state_input, output_gf = self.base_forward(x_screens, x_variables)
        state_input = self.feedforward(state_input)
        output_sc = self.head_forward(state_input)
        return output_sc, output_gf


class DQNFeedforward(DQN):

    DQNModuleClass = DQNModuleFeedforward

    def f_eval(self, last_states):
        screens, variables = self.prepare_f_eval_args(last_states)
        return self.module(
            screens.view(1, -1, *self.screen_shape[1:]),
            [variables[-1, i].view(1) for i in range(self.params.n_variables)]
        )

    def f_train(self, screens, variables, features, actions, rewards, isfinal,
                loss_history=None):

        screens, variables, features, actions, rewards, isfinal = \
            self.prepare_f_train_args(screens, variables, features, actions, rewards, isfinal)

        batch_size = self.params.batch_size
        seq_len = self.hist_size + 1

        screens = screens.view(batch_size, seq_len * self.params.n_fm, *self.screen_shape[1:])

        output_sc1, output_gf1 = self.module(
            screens[:, :-self.params.n_fm, :, :],
            [variables[:, -2, i] for i in range(self.params.n_variables)]
        )
        output_sc2, output_gf2 = self.module(
            screens[:, self.params.n_fm:, :, :],
            [variables[:, -1, i] for i in range(self.params.n_variables)]
        )

        # ---- compute scores (robust to numpy arrays) ----
        device = output_sc1.device

        actions_t = torch.as_tensor(actions, device=device)
        rewards_t = torch.as_tensor(rewards, device=device)
        isfinal_t = torch.as_tensor(isfinal, device=device)

        # Q(s,a) for the taken action (last timestep)
        a_t = actions_t[:, -1].long().view(-1, 1)              # (batch, 1)
        scores1 = output_sc1.gather(1, a_t).squeeze(1)         # (batch,)

        # target: r + gamma * max_a' Q(s',a') * (1 - done)
        with torch.no_grad():
            max_next_q = output_sc2.max(1)[0]                  # (batch,)
            done_f = isfinal_t[:, -1].float()                  # (batch,)
            scores2 = rewards_t[:, -1].float() + (
                self.params.gamma * max_next_q * (1.0 - done_f)
            )

        # dqn loss
        loss_sc = self.loss_fn_sc(scores1, scores2)

        # game features loss
        loss_gf = 0
        if self.n_features:
            loss_gf += self.loss_fn_gf(output_gf1, features[:, -2].float())
            loss_gf += self.loss_fn_gf(output_gf2, features[:, -1].float())

        self.register_loss(loss_history, loss_sc, loss_gf)
        return loss_sc, loss_gf

    @staticmethod
    def validate_params(params):
        DQN.validate_params(params)
        assert params.recurrence == ''