from torch.utils.data import DataLoader

from agents import AgentPPO
from utils import Memory
from utils.ortool import *
from utils.Misc import *
from utils.save_load import *


class TrainPPOAgent:

    def __init__(self,
                 customer_feature,
                 vehicle_feature,
                 customers_count,
                 model_size=128,
                 encoder_layer=3,
                 num_head=8,
                 ff_size_actor=128,
                 ff_size_critic=128,
                 tanh_xplor=10,
                 greedy=False,
                 learning_rate=3e-4,
                 ppo_epoch=3,
                 batch_size=128,
                 entropy_value=0.2,
                 epsilon_clip=0.2,
                 epoch=50,
                 timestep=2,
                 max_grad_norm=2):

        self.greedy = greedy
        self.customer_feature = customer_feature
        self.batch_size = batch_size
        self.customers_count = customers_count
        self.update_timestep = timestep
        self.epoch = epoch
        self.agent = AgentPPO(customer_feature, vehicle_feature, customers_count, model_size,
                              encoder_layer, num_head, ff_size_actor, ff_size_critic,
                              tanh_xplor, greedy, learning_rate,
                              ppo_epoch, batch_size, entropy_value, epsilon_clip, max_grad_norm)

    def run_train(self, args, datas, env, env_params, device, env_test, ref_cost=None):

        if args.resume_state is None:
            start_epoch = 0
        else:
            start_epoch = load_checkpoint(args, self.agent.old_policy, self.agent.optim)

        memory = Memory()
        self.agent.old_policy.to(device)
        print("Running PPO models ")
        train_stats = []
        test_stats = []

        try:
            for epoch in range(start_epoch, args.epoch_count):

                train_data_loader = DataLoader(datas, batch_size=self.batch_size, shuffle=True)
                self.agent.old_policy.train()

                epoch_loss = 0
                epoch_prop = 0
                epoch_val = 0
                epoch_c_val = 0

                with tqdm(train_data_loader, colour='green', desc="Epoch #{: >3d}/{: <3d}".format(epoch + 1, args.epoch_count)) as progress:
                    prop, val, loss_total, loss_a, loss_m, loss_e, norm_r, critic_r = [], [], [], [], [], [], [], []

                    for batch_index, minibatch in enumerate(progress):
                        nodes = minibatch.to(device)
                        nodes = nodes.view(self.batch_size, self.customers_count, 4)
                        dyna_env = env(None, nodes, *env_params)
                        actions, logps, rewards, values = self.agent.old_policy.act(dyna_env)
                        actions = formate_old_actions(actions)
                        actions = torch.tensor(actions)
                        actions = actions.permute(0, 2, 1)

                        actions = actions.to(torch.device('cpu')).detach()
                        logps = logps.to(torch.device('cpu')).detach()
                        rewards = torch.stack(rewards).to(torch.device('cpu')).detach()
                        values = torch.stack([values]).to(torch.device('cpu')).detach()

                        memory.nodes.extend(minibatch)
                        memory.rewards.extend(rewards)
                        memory.values.extend(values)
                        memory.log_probs.extend(logps)
                        memory.actions.extend(actions)

                        if (batch_index + 1) % self.update_timestep == 0:
                            #print('updating part of PPO******************************')
                            loss_total, loss_a, loss_m, loss_e, norm_r, critic_r, ratios, grads = self.agent.update(memory,
                                                                                                                    epoch,
                                                                                                                    env,
                                                                                                                    env_params,
                                                                                                                    device)
                            memory.clear()

                        prob = torch.stack([logps]).sum(dim=0).exp().mean()
                        val = rewards.sum(dim=0).mean()

                        loss_total = torch.tensor(loss_total).mean()
                        loss_a = torch.tensor(loss_a).mean()
                        loss_m = torch.tensor(loss_m).mean()
                        loss_e = torch.tensor(loss_e).mean()
                        norm_r = torch.tensor(norm_r).mean()
                        critic_r = torch.tensor(critic_r).mean()
                        r = torch.tensor(ratios).mean()
                        g = torch.tensor(grads).mean()

                        progress.set_postfix_str("p={:6.4g} val={:6.4g} l_t={:6.4g} l_a={:6.4g} "
                                                 "l_m={:6.4g} l_e={:6.4g} r_n={:6.4g} c_n={:6.4g} r={:6.4g} g={:6.4g} ".format(
                                                  prob.item(), val.item(), loss_total.item(), loss_a.item(),
                                                  loss_m.item(), loss_e.item(), norm_r.item(), critic_r.item(), r.item(), g.item()))

                        epoch_loss += loss_total.item()
                        epoch_prop += prob.item()
                        epoch_val += val.item()
                        epoch_c_val += critic_r.item()

                train_stats.append(stats / args.iter_count for stats in (epoch_loss, epoch_prop, epoch_val, epoch_c_val))
                if ref_cost is not None:
                    test_stats.append(self.test_epoch(self.agent.policy, env_test, ref_cost))

                if args.grad_norm_decay is not None:
                    args.max_grad_norm *= args.grad_norm_decay
                if (epoch + 1) % args.checkpoint_period == 0:
                    save_checkpoint(args, epoch, self.agent.old_policy, self.agent.optim)

        except KeyboardInterrupt:
            save_checkpoint(args, epoch, self.agent.old_policy, self.agent.optim)
        finally:
            export_train_test_stats(args, start_epoch, train_stats, test_stats)

    def test_epoch(self, model, env, ref_costs):
        model.eval()
        costs = env.nodes.new_zeros(env.minibatch)
        with torch.no_grad():
            for _ in range(10):
                _, _, rewards, values = model.act(env)
                costs += torch.stack(rewards).sum(dim=0).squeeze(-1)

        costs = costs / 10
        mean = costs.mean()
        std = costs.std()
        gap = (costs.to(ref_costs.device) / ref_costs - 1).mean()
        print("Cost on test dataset: {:5.2f} +- {:5.2f} ({:.2%})".format(mean, std, gap))
        return mean.item(), std.item(), gap.item()