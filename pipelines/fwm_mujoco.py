import os

import d4rl
import gym
import h5py
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.diffusion import ContinuousRectifiedFlow
from cleandiffuser.invdynamic import FancyMlpInvDynamic
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1d2
from cleandiffuser.utils import DD_RETURN_SCALE, set_seed, loop_dataloader


@hydra.main(config_path="../configs/fwm/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):
    set_seed(args.seed)
    
    return_scale = DD_RETURN_SCALE[args.task.env_name]

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    target_return = args.task.target_return_R2 if args.test_model == "R2" else args.task.target_return_R1
    w_cfg = args.task.w_cfg_R2 if args.test_model == "R2" else args.task.w_cfg_R1


    env = gym.make(args.task.env_name)
    scale = DD_RETURN_SCALE[args.task.env_name]
    dataset = D4RLMuJoCoDataset(
        env.get_dataset(), horizon=args.task.horizon, terminal_penalty=args.terminal_penalty, discount=args.discount)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # =========================== Model Setup ==========================
    fix_mask = torch.zeros((args.task.horizon, obs_dim+1))
    fix_mask[0, :obs_dim] = 1.

    loss_weight = torch.ones((args.task.horizon, obs_dim+1))
    loss_weight[1, :obs_dim] = args.next_obs_loss_weight

    nn_diffusion = DiT1d2(obs_dim+1, emb_dim=args.emb_dim,
              d_model=args.d_model, n_heads=args.n_heads, depth=args.depth, timestep_emb_type="fourier")
    nn_condition = MLPCondition(act_dim, args.emb_dim, hidden_dims=[args.emb_dim, ])

    fwm = ContinuousRectifiedFlow(
            nn_diffusion, nn_condition, fix_mask, loss_weight,
            ema_rate=args.ema_rate, optim_params = {"lr": 3e-4, "weight_decay": 1e-5}, device=args.device)

    # invdyn = FancyMlpInvDynamic(obs_dim, act_dim, 256, nn.Tanh(), add_dropout=True, device=args.device)

    if args.mode == "fwm_training":

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fwm.optimizer, args.diffusion_gradient_steps)

        fwm.train()

        n_gradient_step = 0
        log = dict.fromkeys([f"loss"], 0.)
        for batch in loop_dataloader(dataloader):

            # dd
            # obs = batch["obs"]["state"].to(args.device)
            # act = batch["act"].to(args.device)
            # val = batch["val"].to(args.device) / scale
            
            obs = batch["obs"]["state"].to(args.device)
            rew = batch["rew"].to(args.device)
            act = batch["act"].to(args.device)
            # val = batch["val"].to(args.device) / return_scale     
            trajs = torch.concat([obs, rew], dim=-1)
            # conds = torch.concat([act[:, 0, :], val], dim=-1)       #a_t and RTG as condition

            loss = fwm.update(trajs, act)["loss"]
            log[f"loss"] += loss 
            print(f"{n_gradient_step} loss: {loss}")
            lr_scheduler.step()

            # if n_gradient_step < args.invdyn_gradient_steps:
                # log[f"invdyn_loss"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])["loss"]

            if (n_gradient_step + 1) % args.log_interval == 0:
                log = {k: v / args.log_interval for k, v in log.items()}
                log["gradient_steps"] = n_gradient_step + 1
                print(log)
                log = dict.fromkeys([f"loss" ], 0.)

            if (n_gradient_step + 1) % args.save_interval == 0:
                fwm.save(save_path + f'flow_based_world_model_ckpt_{n_gradient_step + 1}.pt')
                fwm.save(save_path + f'flow_based_world_model_ckpt_latest.pt')
                # if n_gradient_step < args.invdyn_gradient_steps:
                #     invdyn.save(save_path + f'invdyn_ckpt_latest.pt')

            n_gradient_step += 1
            if n_gradient_step > args.diffusion_gradient_steps:
                break

    elif args.mode == "prepare_dataset":

        traj_cond_dataset, condition_dataset, traj_uncond_dataset = [], [], []
        priors = []
        ptr1, ptr2 = 0, 0
        fwm.load(save_path + f'flow_based_world_model_ckpt_{args.reflow_backbone_ckpt}.pt')
        fwm.eval()
        traj_cond_dataset.append(
            torch.zeros((args.cond_dataset_size, 2, args.task.horizon, obs_dim+1), device=args.device))
        condition_dataset.append(
            torch.zeros((args.cond_dataset_size, act_dim), device=args.device))
        traj_uncond_dataset.append(
            torch.zeros((args.uncond_dataset_size, 2, args.task.horizon, obs_dim+1), device=args.device))
        priors.append(
            torch.zeros((args.dataset_prepare_batch_size, args.task.horizon, obs_dim+1), device=args.device))

        dataloader = DataLoader(dataset, batch_size=args.dataset_prepare_batch_size, drop_last=True)

        for batch in loop_dataloader(dataloader):
            for i in range(1):

                obs = batch["obs"]["state"].to(args.device)
                rew = batch["rew"].to(args.device)
                act = batch["act"].to(args.device)
                val = batch["val"].to(args.device) / scale

                priors[i][:, 0, :obs_dim] = obs[:, 0]
                
                if ptr1 < args.cond_dataset_size:
                    noise_cond = torch.randn_like(priors[i])
                    traj_cond, _ = fwm.sample(
                        priors[i], x1=noise_cond, n_samples=obs.shape[0],
                        sample_steps=args.dataset_prepare_sampling_steps, use_ema=True,
                        condition_cfg=act,          #TODO:condition what?
                        w_cfg=1.0, temperature=1.0,
                        sample_step_schedule="quad_continuous")
                    traj_cond_dataset[i][ptr1:ptr1 + obs.shape[0], 0] = traj_cond
                    traj_cond_dataset[i][ptr1:ptr1 + obs.shape[0], 1] = noise_cond
                    condition_dataset[i][ptr1:ptr1 + obs.shape[0]] = val

                if ptr2 < args.uncond_dataset_size:
                    noise_uncond = torch.randn_like(priors[i])
                    traj_uncond, _ = fwm.sample(
                        priors[i], x1=noise_uncond, n_samples=obs.shape[0],
                        sample_steps=args.dataset_prepare_sampling_steps, use_ema=True,
                        condition_cfg=None,
                        w_cfg=0.0, temperature=1.0,
                        sample_step_schedule="quad_continuous")
                    traj_uncond_dataset[i][ptr2:ptr2 + obs.shape[0], 0] = traj_uncond
                    traj_uncond_dataset[i][ptr2:ptr2 + obs.shape[0], 1] = noise_uncond

            if ptr1 < args.cond_dataset_size:
                ptr1 += obs.shape[0]
            if ptr2 < args.uncond_dataset_size:
                ptr2 += obs.shape[0]
            print(
                f'cond: {ptr1 / args.cond_dataset_size * 100.:.1f}%, uncon: {ptr2 / args.uncond_dataset_size * 100.:.1f}%')
            if ptr1 >= args.cond_dataset_size and ptr2 >= args.uncond_dataset_size:
                break

        with h5py.File(save_path + "traj_cond_dataset.h5", "w") as f:
            for i in range(1):
                f.create_dataset(f"reflow_traj_cond_dataset", data=traj_cond_dataset[i].cpu().numpy())
                f.create_dataset(f"reflow_condition_dataset", data=condition_dataset[i].cpu().numpy())
        with h5py.File(save_path + "traj_uncond_dataset.h5", "w") as f:
            for i in range(1):
                f.create_dataset(f"reflow_traj_uncond_dataset", data=traj_uncond_dataset[i].cpu().numpy())

    elif args.mode == "reflow":

        fwm.load(save_path + f'fwm_ckpt_{args.reflow_backbone_ckpt}.pt')
        fwm.train()
        fwm.optimizer.learning_rate = 2e-5

        traj_cond_dataset, traj_uncond_dataset = [], []
        condition_dataset = []
        with h5py.File(save_path + "traj_cond_dataset.h5", "r") as f:
            for i in range(1):
                traj_cond_dataset.append(torch.tensor(f[f"traj_cond_dataset_{i}"][:], device=args.device))
                condition_dataset.append(torch.tensor(f[f"condition_dataset_{i}"][:], device=args.device))
        with h5py.File(save_path + "traj_uncond_dataset.h5", "r") as f:
            for i in range(1):
                traj_uncond_dataset.append(torch.tensor(f[f"traj_uncond_dataset_{i}"][:], device=args.device))

        log = dict.fromkeys([f"loss"], 0.)
        for n_gradient_step in range(args.reflow_gradient_steps):
            for i in range(1):
                if (n_gradient_step % 5) == 0:
                    idx = torch.randint(args.uncond_dataset_size, (args.batch_size,), device=args.device)
                    x01 = traj_uncond_dataset[i][idx]
                    x0, x1 = x01[:, 0], x01[:, 1]
                    log[f"loss{i}"] += fwm.update(x0, x1=x1)["loss"]
                else:
                    idx = torch.randint(args.cond_dataset_size, (args.batch_size,), device=args.device)
                    x01 = traj_cond_dataset[i][idx]
                    val = condition_dataset[i][idx]
                    x0, x1 = x01[:, 0], x01[:, 1]
                    log[f"loss{i}"] += fwm.update(x0, val, x1=x1)["loss"]

            if (n_gradient_step + 1) % args.log_interval == 0:
                log = {k: v / args.log_interval for k, v in log.items()}
                log["gradient_steps"] = n_gradient_step + 1
                print(log)
                log = dict.fromkeys([f"loss"], 0.)

            if (n_gradient_step + 1) % args.save_interval == 0:
                for i in range(1):
                    fwm.save(save_path + f'reflowWM_ckpt_{n_gradient_step + 1}.pt')
                    fwm.save(save_path + f'reflowWM_ckpt_latest.pt')

    elif args.mode == "invdyn_inference":

        fwm.load(save_path + f'{"reflowWM" if args.test_model == "R2" else "fwm"}_ckpt_{args.diffusion_ckpt}.pt')
        fwm.eval()
        invdyn.load(save_path + f'invdyn_ckpt_{args.invdyn_ckpt}.pt')
        invdyn.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        priors = torch.zeros((args.num_envs, args.task.horizon, obs_dim), device=args.device) 
        condition = torch.ones((args.num_envs, 1), device=args.device) * target_return
        for i in range(args.num_episodes):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:

                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                priors[0][:, 0] = obs
                traj, _ = fwm.sample(
                    priors,
                    n_samples=args.num_envs,
                    sample_steps=1 if args.test_model == "R2" else 3, use_ema=args.use_ema,
                    condition_cfg=condition,
                    w_cfg=w_cfg, temperature=args.temperature,
                    sample_step_schedule="quad_continuous", )

                with torch.no_grad():
                    act = invdyn(traj[:, 0], traj[:, 1]).cpu().numpy()

                obs, rew, done, info = env_eval.step(act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
                print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}')

            episode_rewards.append(ep_reward)

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))


if __name__ == "__main__":
    pipeline()
