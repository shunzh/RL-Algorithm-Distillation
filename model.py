import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model


class PolicyTransformerConfig(GPT2Config):
    """
    This class defines the configuration for PolicyTransformer.
    """
    def __init__(
            self,
            hidden_size=128,
            state_dim=0,
            act_dim=0,
            act_num=0,
            max_ep_len=20,
            context_len=80,
            token_mask_prob=0.3,
            **kwargs
    ):
        """
        Args:
            state_dim: the dimensionality of the state space
            act_dim: the dimensionality of the action space
            max_ep_len: the maximum length of an episode

            act_num: the number of actions in the domain (seems to only work for finite action space)
            context_len: the length of the context used as input to the model
            token_mask_prob: the probability of masking a token during training

        for some reason, default values need to be set.
        """
        super().__init__(**kwargs)

        self.hidden_size = hidden_size

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.act_num = act_num
        self.max_ep_len = max_ep_len

        self.context_len = context_len
        self.token_mask_prob = token_mask_prob


class PolicyTransformer(nn.Module):
    """
    A Transformer model that predicts the next action given the history, with a GPT2 backbone.
    Described in In-context Reinforcement Learning with Algorithm Distillation.

    s_0, a_0, r_0,   s_1, a_1, r_1,  ...  s_T, a_T (masked), r_T (masked)

                                           |
                                           V

                                  [logit_1  logit_2  ...  logit_|A|]

                                           |
                                           V

                            Model Output (action with maximum likelihood)
    """
    def __init__(self, config: GPT2Config, *args):
        super().__init__(*args)

        self.config = config

        self.hidden_size = config.hidden_size
        self.encoder = GPT2Model(config)

        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_action = torch.nn.Linear(config.act_dim, config.hidden_size)
        self.embed_reward = torch.nn.Linear(1, config.hidden_size)
        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)

        self.predict_action = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.act_num),
        )

    def forward(
            self,
            states,
            actions,
            rewards,
            timesteps,
            attention_mask = None,
            **kwargs,
    ) -> dict:
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states) # (batch_size, seq_length, hidden_size)
        action_embeddings = self.embed_action(actions) # (batch_size, seq_length, hidden_size)
        reward_embeddings = self.embed_reward(rewards) # (batch_size, seq_length, hidden_size)
        time_embeddings = self.embed_timestep(timesteps) # (batch_size, seq_length, hidden_size)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        reward_embeddings = reward_embeddings + time_embeddings

        # this makes the sequence look like (s_1, a_1, r_1, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack((state_embeddings, action_embeddings, reward_embeddings), dim=1) # (batch_size, 3, seq_length, hidden_size)
            .permute(0, 2, 1, 3) # (batch_size, seq_length, 3, hidden_size)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )

        device = stacked_inputs.device

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1) # (batch_size, 3, seq_length)
            .permute(0, 2, 1) # (batch_size, seq_length, 3)
            .reshape(batch_size, 3 * seq_length) # (batch_size, 3 * seq_length)
        )

        # during training, set stacked_attention_mask to 0 with probability token_mask_prob
        if self.training and self.config.token_mask_prob > 0:
            mask = (torch.rand(stacked_attention_mask.shape) > self.config.token_mask_prob).float().to(device)
            stacked_attention_mask = stacked_attention_mask * mask

        # crucial, always mask the reward and action embeddings of the last time step
        stacked_attention_mask[:, -2:] = 0

        # we feed in the input embeddings (not word indices as in NLP) to the model
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = encoder_outputs[0]

        # reshape x so that the second dimension corresponds to the original
        # states (0), actions (1), or rewards (2); i.e. x[:,0,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3) # (batch_size, 3, seq_length, hidden_size)

        # use the state embedding at the last time step as input, get action distribution
        action_pred = self.predict_action(x[:, 0, -1])  # (batch_size, act_num)

        # get true labels
        # fixme assuming the action space is gym.spaces.Discrete(|A|)
        action_true = actions[:, -1, 0].long()  # (batch_size,), are LongTensor as they serve as labels

        # compute the log probabilities of the predicted actions
        log_probs = F.log_softmax(action_pred, dim=1)
        # compute the NLL loss between the log probabilities and the true actions
        nll_loss = F.nll_loss(log_probs, action_true)

        return {'action_pred': action_pred, 'loss': nll_loss}

    def predict(
            self,
            states=None,
            actions=None,
            rewards=None,
            timesteps=None,
            attention_mask=None,
            temperature=0.0,
            **kwargs,
    ):
        model_output = self.forward(states, actions, rewards, timesteps, attention_mask, **kwargs)

        # choose the action, either by argmax or by sampling from the distribution
        if temperature == 0.:
            action_idx = torch.argmax(model_output['action_pred'], dim=1)
        else:
            action_idx = torch.multinomial(F.softmax(model_output['action_pred'] / temperature, dim=1), num_samples=1)

        return action_idx.item()
