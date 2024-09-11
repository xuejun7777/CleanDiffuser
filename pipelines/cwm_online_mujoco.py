import os
from copy import deepcopy

import d4rl
import gym
import numpy as np
import torch
import hydra
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from collections import namedtuple

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset, D4RLMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader, ReplayBuffer2
from cleandiffuser.diffusion import ContinuousEDM
from cleandiffuser.diffusion.consistency_model import ContinuousConsistencyModel
from cleandiffuser.nn_condition import IdentityCondition, MLPCondition
from cleandiffuser.nn_diffusion import IDQLMlp, DiT1d2, DiT1d
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.utils import (IDQLQNet, IDQLVNet, DD_RETURN_SCALE, DQLCritic, utils, DatasetNormalizer, FreezeModules)
Batch = namedtuple("Batch", "observations actions rewards terminals vals sim_states")
import time

class Timer:

	def __init__(self):
		self._start = time.time()
		self._step_last = 0

	def fps(self, step, reset=True):
		now = time.time()
		fps = (step - self._step_last) / (now - self._start) 
		if reset:
			self._start = now
			self._step_last = step
		return fps


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class OnlineSequenceDataset(torch.utils.data.Dataset):
    """Sequence dataset for online training.

    Requires:
        - prefill_episodes: these are used to compute normalisation constants
    """

    def __init__(
        self,
        prefill_episodes,
        horizon=64,
        normalizer="online_GaussianNormalizer",
        max_path_length=1000,
        max_n_episodes=20000,
        discount=0.99,
        termination_penalty=0,
        use_padding=True,
        norm_keys=["observations", "rewards", "terminals"],
        update_norm_interval=None,
        preprocess_fns=[],
    ):
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.update_norm_interval = update_norm_interval
        self.max_n_episodes = max_n_episodes
        self.termination_penalty = termination_penalty
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.data_buffer = ReplayBuffer2(
            max_n_episodes, max_path_length, termination_penalty
        )
        self.indices = []
        self.initialized = False

        for episode in prefill_episodes:
            self.add_episode(episode)

        # compute and fix normalisation constants based on prefill episodes
        self.normalizer = DatasetNormalizer(self.data_buffer, normalizer)
        self.update_normalizers()

        self.observation_dim = self.data_buffer.observations.shape[-1]
        self.action_dim = self.data_buffer.actions.shape[-1]
        self.n_episodes = self.data_buffer.n_episodes
        self.path_lengths = self.data_buffer.path_lengths
        self.norm_keys = norm_keys

    def reset_data_buffer(self):
        self.data_buffer = ReplayBuffer2(
            self.max_n_episodes, self.max_path_length, self.termination_penalty
        )

    def add_episode(self, episode):
        """Add an episode to the dataset."""
        self.data_buffer.add_path(episode)
        new_episode_num = self.data_buffer.n_episodes - 1
        self.update_indices(new_episode_num)

    def update_normalizers(self):
        self.normalizer.update_statistics(self.data_buffer)

    def get_metrics(self):
        return self.normalizer.get_metrics()

    def update_indices(self, new_episode_num):
        """
        update indices for sampling from dataset to include new episode
        """

        path_length = self.data_buffer.path_lengths[new_episode_num]
        max_start = min(path_length - 1, self.max_path_length - self.horizon)

        max_start = min(path_length - 1, self.max_path_length - self.horizon)
        if not self.use_padding:
            max_start = min(max_start, path_length - self.horizon)

        [
            self.indices.append((new_episode_num, start, start + self.horizon))
            for start in range(max_start)
        ]
        return

    def __len__(self):
        return len(self.indices)

    def get_conditions(self, observations):
        """
        condition on current observation for planning
        """
        return {0: observations[0]}

    def __getitem__(self, _):
        """
        Sample a random batch
        """
        idx = np.random.randint(0, len(self.indices))
        path_ind, start, end = self.indices[idx]

        # trajectory_list = []
        # for key in ["observations", "rewards", "terminals"]:
        #     data = self.data_buffer[key][path_ind, start:end]
        #     if key in self.norm_keys:
        #         data = self.normalizer(data, key)
        #     trajectory_list.append(data)
        
        obs = self.normalizer(self.data_buffer["observations"][path_ind, start:end], "observations") if "observations" in self.norm_keys else self.data_buffer["observations"][path_ind, start:end]
        rewards = self.normalizer(self.data_buffer["rewards"][path_ind, start:end], "rewards") if "rewards" in self.norm_keys else self.data_buffer["rewards"][path_ind, start:end]
        actions = self.data_buffer["actions"][path_ind, start:end]
        terminals = self.normalizer(self.data_buffer["terminals"][path_ind, start:end], "terminals") if "terminals" in self.norm_keys else self.data_buffer["terminals"][path_ind, start:end]
        sim_states = self.data_buffer["sim_states"][path_ind, start:end]

        rtg = self.data_buffer["rewards"][path_ind, start:]
        discounts = self.discounts[:len(rtg)]
        vals = (discounts * rtg).sum()
        vals = np.array([vals], dtype=np.float32)
        
        # trajectories = np.concatenate(trajectory_list, axis=-1)
        batch = Batch(obs, actions, rewards, terminals, vals, sim_states)
        return batch

    def reset(self):
        self.indices = []
    
    def sample_batch(self, batch_size=32):
        # idxs = torch.randperm(len(self))[:batch_size]
        idxs = np.random.choice(len(self), batch_size, replace=False)
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_terminals = []
        batch_vals = []
        batch_sim_states = []

        for idx in idxs:
            data = self[idx]
            batch_obs.append(torch.from_numpy(data.observations))
            batch_actions.append(torch.from_numpy(data.actions))
            batch_rewards.append(torch.from_numpy(data.rewards))
            batch_terminals.append(torch.from_numpy(data.terminals))
            batch_vals.append(torch.from_numpy(data.vals))
            batch_sim_states.append(torch.from_numpy(data.sim_states))
            
        
        
        obs = torch.stack(batch_obs, dim=0)
        actions = torch.stack(batch_actions, dim=0)
        rewards = torch.stack(batch_rewards, dim=0)
        terminals = torch.stack(batch_terminals, dim=0)
        vals = torch.stack(batch_vals, dim=0)  
        sim_states = torch.stack(batch_sim_states, dim=0)
        

        return Batch(obs, actions, rewards, terminals, vals, sim_states)


def cycle(dl):
    while True:
        for data in dl:
            yield data


def get_metrics(
    normalizer,
    obs_norm,
    act_norm,
    rew_norm,
    sim_states,
    max_log=50,):
    metrics = dict()
    obs = normalizer.unnormalize(obs_norm, "observations")
    act = normalizer.unnormalize(act_norm, "actions")
    rew = normalizer.unnormalize(rew_norm, "rewards")
    metrics["data/imag_obs_norm_mean"] = np.mean(obs_norm)
    metrics["data/imag_obs_norm_std"] = np.std(obs_norm)
    metrics["data/best_act_norm_mean"] = np.mean(act_norm)
    metrics["data/best_act_norm_std"] = np.std(act_norm)
    metrics["data/imag_rew_norm_mean"] = np.mean(rew_norm)
    metrics["data/imag_rew_norm_std"] = np.std(rew_norm)
    metrics["data/imag_obs_mean"] = np.mean(obs)
    metrics["data/imag_obs_std"] = np.std(obs)
    metrics["data/best_act_mean"] = np.mean(act)
    metrics["data/best_act_std"] = np.std(act)
    metrics["data/imag_rew_mean"] = np.mean(rew)
    metrics["data/imag_rew_std"] = np.std(rew)

    return metrics

@hydra.main(config_path="../configs/cwm/mujoco", config_name="online_mujoco", version_base=None)
def pipeline(args):    


    seed = args.seed
    device = "cuda:0"
    env_name = args.task.env_name
    return_scale = DD_RETURN_SCALE[args.task.env_name]

    set_seed(seed)
    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
        
    # ---------------------- Create Dataset ----------------------
    env = gym.make(env_name)
    
    random_episodes = utils.random_exploration(args.n_prefill_steps, env)
    
    dataset = OnlineSequenceDataset(random_episodes, horizon=args.task.horizon, termination_penalty=args.termination_penalty)    

    # agent_dataloader = utils.training.cycle(
    #     torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=args.agent_batch_size,
    #         num_workers=2,
    #         shuffle=True,
    #         pin_memory=True,
    #     )
    # )
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    
    wandb.init(project=args.project, name=args.pipeline_name)

    # ------------------------------------------------------------
      

    #CWM
    nn_diffusion = DiT1d2(obs_dim+1, emb_dim=args.emb_dim,
              d_model=args.d_model, n_heads=args.n_heads, depth=args.depth, timestep_emb_type="fourier")
    nn_condition = MLPCondition(act_dim, args.emb_dim, hidden_dims=[args.emb_dim, ])
        
    # ----------------- Masking -------------------
    fix_mask = torch.zeros((args.task.horizon, obs_dim+1))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim+1))
    loss_weight[1, :obs_dim] = args.next_obs_loss_weight
    
    cwm = ContinuousConsistencyModel(
        nn_diffusion, nn_condition, optim_params={"lr": 2e-4},
        fix_mask=fix_mask, loss_weight=loss_weight,
        curriculum_cycle=args.n_environment_steps,
        ema_rate=args.ema_rate, device=device)
    

    cwm_lr_scheduler = CosineAnnealingLR(cwm.optimizer, T_max=args.n_environment_steps)    
    
    cwm.train()
    
    #CPM
    actor_nn_diffusion = DiT1d(act_dim, emb_dim=args.emb_dim,
              d_model=args.d_model, n_heads=args.n_heads, depth=args.depth, timestep_emb_type="fourier")
    actor_nn_condition = MLPCondition(obs_dim+1, args.emb_dim, hidden_dims=[args.emb_dim, ])
        
    actor = ContinuousConsistencyModel(
        actor_nn_diffusion, actor_nn_condition, optim_params={"lr": args.actor_learning_rate},
        curriculum_cycle=args.policy_gradient_steps,
        x_max=+1. * torch.ones((1, args.task.horizon, act_dim)),
        x_min=-1. * torch.ones((1, args.task.horizon, act_dim)),
        ema_rate=args.ema_rate, device=device)
    
    actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.policy_gradient_steps)
    actor.train()
    
    critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)
    critic_target = deepcopy(critic).requires_grad_(False).eval()
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)
    critic.train()
    
    mini_horizon = args.task.mini_horizon
    gammas = torch.tensor([args.discount**i for i in range(mini_horizon)], device=device)
    
    
    def reset_episode():
        done = False
        obs = env.reset()
        episode = {
            "observations": [],
            "actions": [],
            "next_observations": [],
            "rewards": [],
            "terminals": [],
            "sim_states": [],
        }
        t = 0
        return obs, done, episode, t
    


    # ---------------------------- Main Loop ----------------------------------#

    obs, done, episode, t = reset_episode()

    step = 0
    timer = Timer()
    train_metrics_interval = 200
    cwm_n_gradient_step = 0
    imagine_last_log_step = -1
    target_return = torch.ones((args.num_candidates, 1), device=args.device) * args.task.target_return
    log = {"cwm_bc_loss": 0., "cwm_unweighted_bc_loss": 0., "critic_loss": 0., "-actor_loss": 0.}
    action_loss = 0.
    critic_loss = 0.
    while step < args.n_environment_steps:
        metrics = dict()

        # step the policy in the real environment
        # sample actions
        obs = torch.tensor(dataset.normalizer.unnormalize(obs, "observations"), device=args.device, dtype=torch.float32)
        rpt_obs = obs.repeat(args.num_candidates, 1).view(-1, obs_dim)
        cond = torch.concat([rpt_obs, target_return], dim=1)
        act_prior = torch.zeros((args.num_candidates, args.task.horizon, act_dim), device=device)
        act_seq, _ = actor.sample(
            act_prior,
            solver=args.solver,
            n_samples=args.num_candidates,
            sample_steps=args.sampling_steps,
            condition_cfg=cond, w_cfg=1.0,
            use_ema=args.use_ema, temperature=args.temperature)
        act = act_seq[:, 0, :]

        # resample
        with torch.no_grad():
            q = critic_target.q_min(rpt_obs, act)
            q = q.view(args.num_candidates, 1)
            w = torch.softmax(q * args.task.weight_temperature, 1)
            act = act.view(args.num_candidates, act_dim)
            index = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
            sampled_act = act[index].cpu().numpy()
        
        next_obs, rew, done, info = env.step(sampled_act)
        # done = term or trunc
        t += 1

        episode["observations"].append(obs.cpu().detach().numpy().copy())
        episode["actions"].append(sampled_act.copy())
        episode["next_observations"].append(next_obs.copy())
        episode["rewards"].append(rew.copy())
        episode["terminals"].append(done)
        if "sim_state" in info.keys():
            episode["sim_states"].append(info["sim_state"].copy())
        else:
            episode["sim_states"].append(None)

        obs = next_obs
        if done or t >= args.max_path_length:
            episode = {key: np.array(episode[key]) for key in episode.keys()}
            episode["timeouts"] = np.array([False] * len(episode["rewards"]))
            ret = np.sum(episode["rewards"])
            print("Episode Return: ", ret, "Length: ", len(episode["rewards"]))
            metrics.update({"expl/return": ret, "expl/length": len(episode["rewards"])})
            dataset.add_episode(episode)
            state, done, episode, t = reset_episode()

            if args.update_normalization:
                dataset.update_normalizers()

        if step % int(1 / args.train_agent_ratio) == 0:
            #policy training   TODO: add the policy gradient update?????
            cwm.eval()
            if step >= args.pretrain_CWM:
                
                batch = dataset.sample_batch(args.batch_size)
                batch_obs = batch.observations.to(device)
                batch_act = batch.actions.to(device)
                batch_rew = batch.rewards.to(device).squeeze()
                batch_val = batch.vals.to(device) / return_scale
                batch_tml = batch.terminals.to(device)
                
                # trajs = torch.concat([batch_obs, batch_rew], dim=-1)
                actor_conds = torch.concat([batch_obs[:, 0, :], batch_val], dim=-1)
                
                # priors = torch.zeros((args.batch_size, args.task.horizon, obs_dim+1), device=args.device)
                # priors[:, 0, :obs_dim] = batch_obs[:, 0]
                
                # trajs, log_s = cwm.sample(
                #     priors,
                #     n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=args.use_ema,
                #     condition_cfg=batch_act, w_cfg=args.task.w_cfg, temperature=args.temperature, requires_grad=True)

                # imagine_obs = trajs[:, :, :obs_dim]        # [batch_size, horizon, obs_dim] same as obs
                # imagine_rew = trajs[:, :, -1]
                # imagine_tml = dataset.normalizer.unnormalize(trajs[:, :, -1], "terminals")
                # imagine_tml = torch.where(imagine_tml>0.5, 1.0, 0.0)
                
                #TODO: maybe we can calculate the error between the imagine and the real data.
                
                # Critic Training using real data TODO: use imagine data or real data?????
                current_q1, current_q2 = critic(batch_obs[:, :args.task.horizon-mini_horizon], batch_act[:, :args.task.horizon-mini_horizon])
                prior = torch.zeros((args.batch_size, args.task.horizon, act_dim), device=device)
                act_seq, _ = actor.sample(
                        prior, solver=args.solver,
                        n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=True,
                        temperature=1.0, condition_cfg=actor_conds, w_cfg=1.0, requires_grad=False)
                # not_done_list = [torch.concat([torch.ones((args.batch_size, 1), device=args.device), 1 - imagine_tml[:, i:i+mini_horizon-1]], dim=1) for i in range(args.task.horizon-mini_horizon)]    #have decided to use termination_penalty TODO: Do we really need to be so cumbersome????
                cum_rews = torch.concat([torch.sum(gammas *  batch_rew[:, i:i+mini_horizon], dim=-1, keepdim=True) for i in range(args.task.horizon-mini_horizon)], dim=1)
                # target_qs = (cum_rews.unsqueeze(2) + args.discount**mini_horizon * (1 - imagine_tml[:, mini_horizon-1:-1]) * torch.min(*critic_target(imagine_obs[:, mini_horizon:], act_seq[:, mini_horizon:]))).detach()
                target_qs = (cum_rews.unsqueeze(2) + args.discount**mini_horizon * torch.min(*critic_target(batch_obs[:, mini_horizon:], act_seq[:, mini_horizon:]))).detach()

                critic_loss = F.mse_loss(current_q1, target_qs) + F.mse_loss(current_q2, target_qs)

                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()
                
                # act_prior = torch.zeros((args.batch_size, args.task.horizon, act_dim), device=device)
                # act4PG, _ = actor.sample(
                #         act_prior, solver=args.solver,
                #         n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=False,
                #         temperature=1.0, condition_cfg=actor_conds, w_cfg=1.0, requires_grad=True)
                
                # with FreezeModules([critic, ]):
                #     q1_new_action, q2_new_action = critic(batch_obs, act4PG)
                # if np.random.uniform() > 0.5:
                #     q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
                # else:
                #     q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
                # q_loss.backward()  #another policy improvement method
                
                actions_optim = torch.optim.Adam([batch_act], lr=args.policy_gradient_lr, eps=1e-5)

                priors = torch.zeros((args.batch_size, args.task.horizon, obs_dim+1), device=args.device)
                priors[:, 0, :obs_dim] = batch_obs[:, 0, :]
                for i in range(args.policy_optimalization_steps+1):
                    batch_act.requires_grad_(True)
                    trajs, log_s = cwm.sample(
                        priors, 
                        n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=args.use_ema,
                        condition_cfg=batch_act, w_cfg=args.task.w_cfg, temperature=args.temperature, requires_grad=True)
                    
                    imagine_obs = trajs[:, :, :obs_dim]
                    imagine_rew = trajs[:, :, -1]
                    # imagine_tml = trajs[:, :, -1]
                    # not_done_list = [torch.concat([torch.ones((args.batch_size, 1), device=args.device), 1 - imagine_tml[:, i:i+mini_horizon-1]], dim=1) for i in range(args.task.horizon-mini_horizon)]    #have decided to use termination_penalty TODO: Do we really need to be so cumbersome????
                    cum_rew =  torch.sum(gammas * imagine_rew[:, :mini_horizon], dim=-1)
                    action_loss = -(cum_rew.unsqueeze(1) + args.discount**mini_horizon * torch.min(*critic(imagine_obs[:, mini_horizon], batch_act[:, mini_horizon])))

                    if i == args.policy_optimalization_steps:
                        break
                    
                    actions_optim.zero_grad()

                    action_loss.backward(torch.ones_like(action_loss))

                    actions_optim.step()

                    batch_act.requires_grad_(False)
                    batch_act.clamp_(-1., 1.)

                
                batch_act = batch_act.detach()           #FIXME: Do we need to replace new data in dataset?
                actor_conds[:, -1] = -action_loss.squeeze().detach() / return_scale
                agent_metrics = actor.update(batch_act, actor_conds, update_ema=False)  #DONE FIXME: actor_conds with new RTG

                actor_lr_scheduler.step()
                
                # -- ema
                if step % args.ema_update_interval == 0:
                    if step >= args.ema_update_interval:
                        actor.ema_update()
                    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                        target_param.data.copy_(args.ema_rate * param.data + (1 - args.ema_rate) * target_param.data)
                
                if step >= imagine_last_log_step + args.log_interval/10:
                    metrics.update(
                        get_metrics(
                            dataset.normalizer,
                            imagine_obs.cpu().detach().numpy(),
                            batch_act.cpu().detach().numpy(),
                            imagine_rew.cpu().detach().numpy(),
                            step,
                            max_log=50,
                        )
                    )
                    imagine_last_log_step = step
                
                if step % train_metrics_interval == 0:
                    [
                        metrics.update({f"cosistency_agent/{key}": agent_metrics[key]})
                        for key in agent_metrics.keys()
                    ]
                    metrics.update({f"cosistency_agent/critic_loss": critic_loss})
                    metrics.update({f"cosistency_agent/-action_loss": -action_loss})

            log["-actor_loss"] += -action_loss
            log["critic_loss"] += critic_loss
            consistency_updates = int(args.train_onsistency_ratio / args.train_agent_ratio)
            cwm.train()
            for _ in range(consistency_updates):
                
                batch = dataset.sample_batch(args.batch_size)
                _obs = batch.observations.to(device)
                _act = batch.actions.to(device)
                _rew = batch.rewards.to(device)
                # _tml = batch.terminals.to(device)
                
                trajs = torch.concat([_obs, _rew], dim=-1)
                
                # -- world model Training
                world_model_metrics = cwm.update(trajs, _act)
                loss = world_model_metrics["loss"]
                
                cwm_lr_scheduler.step()
                
                # print(f"step: {n_gradient_step + 1}, loss: {loss}")

            # ----------- Logging ------------
                log["cwm_bc_loss"] += loss
                log["cwm_unweighted_bc_loss"] += world_model_metrics["unweighted_loss"]

                if (cwm_n_gradient_step + 1) % args.log_interval == 0:
                    log["gradient_steps"] = cwm_n_gradient_step + 1
                    log["cwm_bc_loss"] /= args.log_interval
                    log["cwm_unweighted_bc_loss"] /= args.log_interval
                    log["-actor_loss"] /= args.log_interval
                    log["critic_loss"] /= args.log_interval
                    log["curriculum_process"] = cwm.cur_logger.curriculum_process
                    log["Nk"] = cwm.cur_logger.Nk
                    print(log)
                    log = {"cwm_bc_loss": 0., "cwm_unweighted_bc_loss": 0., "critic_loss": 0., "-actor_loss": 0.}
                cwm_n_gradient_step += 1

                
            if step % train_metrics_interval == 0:
                [
                    metrics.update({f"CWM/{key}": world_model_metrics[key]})
                    for key in world_model_metrics.keys()
                ]
                
                

        if step % args.log_interval == 0:
            dataset_metrics = dataset.get_metrics()
            [
                metrics.update({f"dataset/{key}": dataset_metrics[key]})
                for key in dataset_metrics.keys()
            ]
            metrics.update({"fps": timer.fps(step)})

        # ----------- Saving ------------
        if (step + 1) % args.save_interval == 0:
            actor.save(save_path + f"policy_ckpt_latest.pt")
            actor.save(save_path + f"policy_ckpt_{step + 1}.pt")
            torch.save({
                "critic": critic.state_dict(),
                "critic_target": critic_target.state_dict(),
            }, save_path + f"critic_ckpt_latest.pt")


        # if step % args.eval_interval == 0:
        #     eval_metrics = evaluate_policy(
        #         ac.forward_actor,
        #         eval_env,
        #         device,
        #         step,
        #         dataset,
        #         use_mean=True,
        #         n_episodes=20,
        #         renderer=renderer,
        #     )
        # torch.cuda.empty_cache()
        wandb.log(metrics, step=step)
        step += 1
    
    
    
        
            
if __name__ == "__main__":
    pipeline()
