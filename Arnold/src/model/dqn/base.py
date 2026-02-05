import numpy as np
import torch
import torch.nn as nn
from logging import getLogger

from ...utils import bool_flag
from ..utils import value_loss, build_CNN_network
from ..utils import build_game_variables_network, build_game_features_network


logger = getLogger()


class DQNModuleBase(nn.Module):

    def __init__(self, params):
        super(DQNModuleBase, self).__init__()

        # build CNN network
        build_CNN_network(self, params)
        self.output_dim = self.conv_output_dim

        # game variables network
        build_game_variables_network(self, params)
        if self.n_variables:
            self.output_dim += sum(params.variable_dim)

        # dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)

        # game features network
        build_game_features_network(self, params)

        # Estimate state-action value function Q(s, a)
        # If dueling network, estimate advantage function A(s, a)
        self.proj_action_scores = nn.Linear(params.hidden_dim, self.n_actions)

        self.dueling_network = params.dueling_network
        if self.dueling_network:
            self.proj_state_values = nn.Linear(params.hidden_dim, 1)

        # log hidden layer sizes
        logger.info('Conv layer output dim : %i' % self.conv_output_dim)
        logger.info('Hidden layer input dim: %i' % self.output_dim)

    def base_forward(self, x_screens, x_variables):
        """
        Argument sizes:
            - x_screens of shape (batch_size, conv_input_size, h, w)
            - x_variables of shape (batch_size,)
        """
        batch_size = x_screens.size(0)

        # convolution
        x_screens = x_screens / 255.
        conv_output = self.conv(x_screens).view(batch_size, -1)

        # game variables
        if self.n_variables:
            embeddings = [self.game_variable_embeddings[i](x_variables[i])
                          for i in range(self.n_variables)]

        # game features
        if self.n_features:
            output_gf = self.proj_game_features(conv_output)
        else:
            output_gf = None

        # create state input
        if self.n_variables:
            output = torch.cat([conv_output] + embeddings, 1)
        else:
            output = conv_output

        # dropout
        if self.dropout:
            output = self.dropout_layer(output)

        return output, output_gf

    def head_forward(self, state_input):
        if self.dueling_network:
            a = self.proj_action_scores(state_input)  # advantage branch
            v = self.proj_state_values(state_input)   # state value branch
            a -= a.mean(1, keepdim=True).expand(a.size())
            return v.expand(a.size()) + a
        else:
            return self.proj_action_scores(state_input)


class DQN(object):

    def __init__(self, params):
        # network parameters
        self.params = params
        self.screen_shape = (params.n_fm, params.height, params.width)
        self.hist_size = params.hist_size
        self.n_variables = params.n_variables
        self.n_features = params.n_features

        # main module + loss functions
        self.module = self.DQNModuleClass(params)
        self.loss_fn_sc = value_loss(params.clip_delta)
        self.loss_fn_gf = nn.BCELoss()

        # cuda
        self.cuda = params.gpu_id >= 0
        if self.cuda:
            self.module.cuda()

    def get_var(self, x: torch.Tensor) -> torch.Tensor:
        """Move a tensor to a CPU / GPU tensor."""
        return x.cuda() if self.cuda else x

    def reset(self):
        pass

    def new_loss_history(self):
        return dict(dqn_loss=[], gf_loss=[])

    def log_loss(self, loss_history):
        logger.info('DQN loss: %.5f' % np.mean(loss_history['dqn_loss']))
        if self.n_features > 0:
            logger.info('Game features loss: %.5f' % np.mean(loss_history['gf_loss']))

    def prepare_f_eval_args(self, last_states):
        """
        Prepare inputs for evaluation.
        """
        screens = np.float32([s.screen for s in last_states])
        screens = self.get_var(torch.tensor(screens, dtype=torch.float32))
        assert screens.size() == (self.hist_size,) + self.screen_shape

        if self.n_variables:
            variables = np.int64([s.variables for s in last_states])
            variables = self.get_var(torch.tensor(variables, dtype=torch.long))
            assert variables.size() == (self.hist_size, self.n_variables)
        else:
            variables = None

        return screens, variables

    def prepare_f_train_args(self, screens, variables, features,
                             actions, rewards, isfinal):
        """
        Prepare inputs for training.
        """
        # Convert numpy -> torch tensors (old code used Variable + .copy)
        screens = self.get_var(torch.tensor(np.asarray(screens, dtype=np.float32), dtype=torch.float32))
        if self.n_variables:
            variables = self.get_var(torch.tensor(np.asarray(variables, dtype=np.int64), dtype=torch.long))
        if self.n_features:
            features = self.get_var(torch.tensor(np.asarray(features, dtype=np.int64), dtype=torch.long))

        # IMPORTANT: make actions a torch tensor too (many downstream ops assume torch)
        actions = torch.tensor(np.asarray(actions, dtype=np.int64), dtype=torch.long)
        actions = self.get_var(actions)

        rewards = self.get_var(torch.tensor(np.asarray(rewards, dtype=np.float32), dtype=torch.float32))
        isfinal = self.get_var(torch.tensor(np.asarray(isfinal, dtype=np.float32), dtype=torch.float32))

        recurrence = self.params.recurrence
        batch_size = self.params.batch_size
        n_updates = 1 if recurrence == '' else self.params.n_rec_updates
        seq_len = self.hist_size + n_updates

        # check tensors sizes
        assert screens.size() == (batch_size, seq_len) + self.screen_shape
        if self.n_variables:
            assert variables.size() == (batch_size, seq_len, self.n_variables)
        if self.n_features:
            assert features.size() == (batch_size, seq_len, self.n_features)
        assert actions.size() == (batch_size, seq_len - 1)
        assert rewards.size() == (batch_size, seq_len - 1)
        assert isfinal.size() == (batch_size, seq_len - 1)

        return screens, variables, features, actions, rewards, isfinal

    def register_loss(self, loss_history, loss_sc, loss_gf):
        # modern scalar extraction
        loss_history['dqn_loss'].append(float(loss_sc.item()))
        loss_history['gf_loss'].append(float(loss_gf.item()) if self.n_features else 0.0)

    def next_action(self, last_states, save_graph=False):
        scores, pred_features = self.f_eval(last_states)
        if self.params.network_type == 'dqn_ff':
            assert scores.size() == (1, self.module.n_actions)
            scores = scores[0]
            if pred_features is not None:
                assert pred_features.size() == (1, self.module.n_features)
                pred_features = pred_features[0]
        else:
            assert self.params.network_type == 'dqn_rnn'
            seq_len = 1 if self.params.remember else self.params.hist_size
            assert scores.size() == (1, seq_len, self.module.n_actions)
            scores = scores[0, -1]
            if pred_features is not None:
                assert pred_features.size() == (1, seq_len, self.module.n_features)
                pred_features = pred_features[0, -1]

        # modern argmax
        action_id = int(scores.argmax(dim=0).item())
        self.pred_features = pred_features
        return action_id

    @staticmethod
    def register_args(parser):
        # batch size / replay memory size
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
        parser.add_argument("--replay_memory_size", type=int, default=1000000, help="Replay memory size")

        # epsilon decay
        parser.add_argument("--start_decay", type=int, default=0, help="Learning step when the epsilon decay starts")
        parser.add_argument("--stop_decay", type=int, default=1000000, help="Learning step when the epsilon decay stops")
        parser.add_argument("--final_decay", type=float, default=0.1, help="Epsilon value after decay")

        # discount factor / dueling architecture
        parser.add_argument("--gamma", type=float, default=0.99, help="Gamma")
        parser.add_argument("--dueling_network", type=bool_flag, default=False,
                            help="Use a dueling network architecture")

        # recurrence type
        parser.add_argument("--recurrence", type=str, default='', help="Recurrent neural network (RNN / GRU / LSTM)")

    @staticmethod
    def validate_params(params):
        assert 0 <= params.start_decay <= params.stop_decay
        assert 0 <= params.final_decay <= 1
        assert params.replay_memory_size >= 1000