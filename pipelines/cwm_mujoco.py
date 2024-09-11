import os
from copy import deepcopy

import d4rl
import gym
import numpy as np
import torch
import hydra
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset, D4RLMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import ContinuousEDM
from cleandiffuser.diffusion.consistency_model import ContinuousConsistencyModel
from cleandiffuser.nn_condition import IdentityCondition, MLPCondition
from cleandiffuser.nn_diffusion import IDQLMlp, DiT1d2, DiT1d
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.utils import (IDQLQNet, IDQLVNet, DD_RETURN_SCALE, DQLCritic)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    

@hydra.main(config_path="../configs/cwm/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):    


    seed = args.seed
    device = "cuda:0"
    env_name = args.task.env_name
    weight_temperature = 100. # 10 for me / 100 for m / 400 for mr
    return_scale = DD_RETURN_SCALE[args.task.env_name]

    set_seed(seed)
    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
        
    # ---------------------- Create Dataset ----------------------
    env = gym.make(env_name)
    
    # TODO: few terminal states, do we need change the dataset class?
    
    dataset = D4RLMuJoCoDataset(
        env.get_dataset(), horizon=args.task.horizon, terminal_penalty=args.terminal_penalty, discount=args.discount)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # ------------------------------------------------------------
      
    """ 
    Train the Consistency Model without relying on any pre-trained Models.
    """
    
    # As suggested in "IMPROVED TECHNIQUES FOR TRAINING CONSISTENCY MODELS", the Fourier scale is set to 0.02 instead of default 16.0.
    # nn_diffusion = DiT1d(
    #     obs_dim+1, emb_dim=args.emb_dim,
    #     d_model=args.d_model, n_heads=args.n_heads, depth=args.depth, timestep_emb_type="fourier")
    # nn_condition = MLPCondition(in_dim=act_dim+1, out_dim=args.emb_dim, hidden_dims=[args.emb_dim, ], act=nn.SiLU(), dropout=args.label_dropout)
    
    #CWM training
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
        curriculum_cycle=args.diffusion_gradient_steps,
        ema_rate=args.ema_rate, device=device)
    

    cwm_lr_scheduler = CosineAnnealingLR(cwm.optimizer, T_max=args.diffusion_gradient_steps)    
    
    if args.mode == "cwm_invdyn_training":
        
        invdyn = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=device)

        invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps)
        
        cwm.train()  

        n_gradient_step = 0
        log = {"bc_loss": 0., "unweighted_bc_loss": 0., "avg_loss_invdyn":0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(device)
            rew = batch["rew"].to(device)
            act = batch["act"].to(device)
            val = batch["val"].to(device) / return_scale
            trajs = torch.concat([obs, rew], dim=-1)
            conds = torch.concat([act[:, 0, :], val], dim=-1)
            
            # -- world model Training
            _log = cwm.update(trajs, conds)

            cwm_lr_scheduler.step()
            
            if n_gradient_step <= args.invdyn_gradient_steps:
                log["avg_loss_invdyn"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])['loss']
                invdyn_lr_scheduler.step()
            
            print(_log["loss"])
            # ----------- Logging ------------
            log["bc_loss"] += _log["loss"]
            log["unweighted_bc_loss"] += _log["unweighted_loss"]

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["bc_loss"] /= args.log_interval
                log["unweighted_bc_loss"] /= args.log_interval
                log["avg_loss_invdyn"] /= args.log_interval
                log["curriculum_process"] = cwm.cur_logger.curriculum_process
                log["Nk"] = cwm.cur_logger.Nk
                print(log)
                log = {"bc_loss": 0., "unweighted_bc_loss": 0., "avg_loss_invdyn":0.}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                cwm.save(save_path + f"cwm_ckpt_latest.pt")
                cwm.save(save_path + f"cwm_ckpt_{n_gradient_step + 1}.pt")
                invdyn.save(save_path + f"invdyn_ckpt_latest.pt")
                invdyn.save(save_path + f"invdyn_ckpt_{n_gradient_step + 1}.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break
    elif args.mode == "cwm_training":
        
        cwm.train()  

        n_gradient_step = 0
        log = {"bc_loss": 0., "unweighted_bc_loss": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(device)
            rew = batch["rew"].to(device)
            act = batch["act"].to(device)
            # val = batch["val"].to(device) / return_scale
            
            trajs = torch.concat([obs, rew], dim=-1)
            # conds = torch.concat([act[:, 0, :], val], dim=-1)
            
            # -- world model Training
            _log = cwm.update(trajs, act)
            loss = _log["loss"]
            
            cwm_lr_scheduler.step()
            
            # print(f"step: {n_gradient_step + 1}, loss: {loss}")
            # ----------- Logging ------------
            log["bc_loss"] += loss
            log["unweighted_bc_loss"] += _log["unweighted_loss"]

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["bc_loss"] /= args.log_interval
                log["unweighted_bc_loss"] /= args.log_interval
                log["curriculum_process"] = cwm.cur_logger.curriculum_process
                log["Nk"] = cwm.cur_logger.Nk
                print(log)
                log = {"bc_loss": 0., "unweighted_bc_loss": 0.}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                cwm.save(save_path + f"cwm_ckpt_latest.pt")
                cwm.save(save_path + f"cwm_ckpt_{n_gradient_step + 1}.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break
    elif args.mode == "invdyn_inference(dd)":
        cwm.load(save_path + f"cwm_ckpt_{args.diffusion_ckpt}.pt")
        cwm.eval()
        invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
        invdyn.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim+1), device=args.device)
        condition = torch.ones((args.num_envs, act_dim+1), device=args.device) * args.task.target_return
        for i in range(args.num_episodes):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:
                # normalize obs
                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                # sample trajectories
                prior[:, 0, :obs_dim] = obs
                traj, log = cwm.sample(
                    prior, 
                    n_samples=args.num_envs, sample_steps=args.sampling_steps, use_ema=args.use_ema,
                    condition_cfg=condition, w_cfg=args.task.w_cfg, temperature=args.temperature)

                # inverse dynamic
                with torch.no_grad():
                    act = invdyn.predict(obs, traj[:, 1, :]).cpu().numpy()

                # step
                obs, rew, done, info = env_eval.step(act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
                print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}')

            episode_rewards.append(ep_reward)

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))
        
    elif args.mode == "rl_training_with_cwm":
        pass
    elif args.mode == "policy_training_with_cwm_byGD":
        '''
        policy module can be MLP or diffusion, flow, consistency model
        '''
        # dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), True)        
        cwm.load(save_path + f"cwm_ckpt_{args.diffusion_ckpt}.pt")
        cwm.eval()
        
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
        n_gradient_step = 0
        log = {"policy_bc_loss": 0., "policy_unweighted_bc_loss": 0.,  "critic_loss": 0., "-actor_loss": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(device)
            rew = batch["rew"].to(device)
            act = batch["act"].to(device)
            val = batch["val"].to(device) / return_scale
            
            trajs = torch.concat([obs, rew], dim=-1)
            actor_conds = torch.concat([obs[:, 0, :], val], dim=-1)
            
            priors = torch.zeros((args.batch_size, args.task.horizon, obs_dim+1), device=args.device)
            priors[:, 0, :obs_dim] = obs[:, 0]
            
            trajs, log_s = cwm.sample(
                priors,
                n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=args.use_ema,
                condition_cfg=act, w_cfg=args.task.w_cfg, temperature=args.temperature, requires_grad=True)

            imagine_obs = trajs[:, :, :obs_dim]        # [batch_size, horizon, obs_dim] same as obs
            imagine_rew = trajs[:, :, -1]
            
            # Critic Training
            current_q1, current_q2 = critic(obs[:, :args.task.horizon-mini_horizon], act[:, :args.task.horizon-mini_horizon])
            prior = torch.zeros((args.batch_size, args.task.horizon, act_dim), device=device)
            act_seq, _ = actor.sample(
                    prior, solver=args.solver,
                    n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=True,
                    temperature=1.0, condition_cfg=actor_conds, w_cfg=1.0, requires_grad=False)

            cum_rews = torch.concat([torch.sum(gammas * imagine_rew[:, i:i+mini_horizon], dim=-1, keepdim=True) for i in range(args.task.horizon-mini_horizon)], dim=1)
            
            target_qs = (cum_rews.unsqueeze(2) + args.discount**mini_horizon * torch.min(*critic_target(imagine_obs[:, mini_horizon:], act_seq[:, mini_horizon:]))).detach()

            critic_loss = F.mse_loss(current_q1, target_qs) + F.mse_loss(current_q2, target_qs)

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()
            
            # -- Policy Training
            
            # act_prior = torch.zeros((args.batch_size, args.task.horizon, act_dim), device=device)
            # best_actions, _ = actor.sample(
            #         act_prior, solver=args.solver,
            #         n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=True,
            #         temperature=1.0, condition_cfg=actor_conds, w_cfg=1.0, requires_grad=True)
            
            best_actions = act
            actions_optim = torch.optim.Adam([best_actions], lr=args.policy_gradient_lr, eps=1e-5)

            priors = torch.zeros((args.batch_size, args.task.horizon, obs_dim+1), device=args.device)
            priors[:, 0, :obs_dim] = obs[:, 0, :]
            for i in range(args.policy_optimalization_steps+1):
                best_actions.requires_grad_(True)
                trajs, log_s = cwm.sample(
                    priors, 
                    n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=args.use_ema,
                    condition_cfg=best_actions, w_cfg=args.task.w_cfg, temperature=args.temperature, requires_grad=True)
                
                imagine_obs = trajs[:, :, :obs_dim]
                imagine_rew = trajs[:, :, -1]
                cum_rew =  torch.sum(gammas * imagine_rew[:, :mini_horizon], dim=-1)
                action_loss = -(cum_rew.unsqueeze(1) + args.discount**mini_horizon * torch.min(*critic (imagine_obs[:, mini_horizon], best_actions[:, mini_horizon])))

                if i == args.policy_optimalization_steps:
                    break
                
                actions_optim.zero_grad()

                action_loss.backward(torch.ones_like(action_loss))

                actions_optim.step()

                best_actions.requires_grad_(False)
                best_actions.clamp_(-1., 1.)

            

            best_actions = best_actions.detach()
            actor_conds[:, -1] = -action_loss.squeeze().detach() / return_scale
            _log = actor.update(best_actions, actor_conds)  #DONE FIXME: actor_conds with new RTG 
            loss = _log["loss"]

            actor_lr_scheduler.step()
            
            # -- ema
            if n_gradient_step % args.ema_update_interval == 0:
                if n_gradient_step >= 1000:
                    actor.ema_update()
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(args.ema_rate * param.data + (1 - args.ema_rate) * target_param.data)
            
            # print(f"step: {n_gradient_step + 1}, loss: {loss}")
            # ----------- Logging ------------
            log["policy_bc_loss"] += loss
            log["policy_unweighted_bc_loss"] += _log["unweighted_loss"]
            log["critic_loss"] += critic_loss.item()
            log["-actor_loss"] += -action_loss.mean().item()
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["policy_bc_loss"] /= args.log_interval
                log["policy_unweighted_bc_loss"] /= args.log_interval
                log["critic_loss"] /= args.log_interval
                log["-actor_loss"] /= args.log_interval
                log["curriculum_process"] = cwm.cur_logger.curriculum_process
                log["Nk"] = cwm.cur_logger.Nk
                print(log)
                log = {"policy_bc_loss": 0., "policy_unweighted_bc_loss": 0.,  "critic_loss": 0., "-actor_loss": 0.}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                actor.save(save_path + f"policy_ckpt_latest.pt")
                actor.save(save_path + f"policy_ckpt_{n_gradient_step + 1}.pt")
                torch.save({
                    "critic": critic.state_dict(),
                    "critic_target": critic_target.state_dict(),
                }, save_path + f"critic_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.policy_gradient_steps:
                break
    
    elif args.mode == "inference":
        actor_nn_diffusion = DiT1d(act_dim, emb_dim=args.emb_dim,
              d_model=args.d_model, n_heads=args.n_heads, depth=args.depth, timestep_emb_type="fourier")
        actor_nn_condition = MLPCondition(obs_dim+1, args.emb_dim, hidden_dims=[args.emb_dim, ])
            
        actor = ContinuousConsistencyModel(
            actor_nn_diffusion, actor_nn_condition, optim_params={"lr": args.actor_learning_rate},
            curriculum_cycle=args.policy_gradient_steps,
            x_max=+1. * torch.ones((1, args.task.horizon, act_dim)),
            x_min=-1. * torch.ones((1, args.task.horizon, act_dim)),
            ema_rate=args.ema_rate, device=device)
        
        critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)
        critic_target = deepcopy(critic).requires_grad_(False).eval()
        
        actor.load(save_path + f"policy_ckpt_{args.actor_ckpt}.pt")
        critic_ckpt = torch.load(save_path + f"critic_ckpt_{args.critic_ckpt}.pt")
        critic.load_state_dict(critic_ckpt["critic"])
        critic_target.load_state_dict(critic_ckpt["critic_target"])

        actor.eval()
        critic.eval()
        critic_target.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((args.num_envs * args.num_candidates, args.task.horizon, act_dim), device=args.device)
        target_return = torch.ones((args.num_envs, 1), device=args.device) * args.task.target_return
        for i in range(args.num_episodes):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:
                # normalize obs
                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                obs = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)
                cond = torch.concat([obs, target_return.repeat(args.num_candidates, 1)], 1)

                # sample actions
                act_seq, log = actor.sample(
                    prior,
                    solver=args.solver,
                    n_samples=args.num_envs * args.num_candidates,
                    sample_steps=args.sampling_steps,
                    condition_cfg=cond, w_cfg=1.0,
                    use_ema=args.use_ema, temperature=args.temperature)
                act = act_seq[:, 0, :]

                # resample
                with torch.no_grad():
                    q = critic_target.q_min(obs, act)
                    q = q.view(-1, args.num_candidates, 1)
                    w = torch.softmax(q * args.task.weight_temperature, 1)
                    act = act.view(-1, args.num_candidates, act_dim)

                    indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
                    sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()
                    
                print(sampled_act.shape, sampled_act)

                # step
                obs, rew, done, info = env_eval.step(sampled_act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
                print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}')

                if np.all(cum_done):
                    break

            episode_rewards.append(ep_reward)

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))

    else:
        raise ValueError(f"Invalid mode: {args.mode}")
        
            
if __name__ == "__main__":
    pipeline()


# metrics = dict()
# while step < args.steps_policytrainig:

#     agent = Agent(obs_dim, act_dim, dataset.get_normalizer, hidden_dim=args.emb_dim)

#     agent_metrics = agent.multistep_training(
#         imagine_states, actions, imagine_reward, step
#     )
    
#     if step % args.train_metrics_interval(200) == 0:
#         [
#             metrics.update({f"agent/{key}": agent_metrics[key]})
#             for key in agent_metrics.keys()
#         ]

#     wandb.log(metrics, step=step)
#     step += 1