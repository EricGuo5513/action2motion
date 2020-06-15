import os
import time
import math
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import torch.functional as F
from utils.utils_ import print_current_loss
from lie.pose_lie import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


class Trainer(object):
    def __init__(self, pose_sampler, motion_sampler, opt, device):
        self.opt = opt
        self.device = device
        self.pose_sampler = pose_sampler
        self.motion_sampler = motion_sampler
        self.pose_enumerator = None
        self.motion_enumerator = None
        if opt.isTrain:
            self.use_categories = opt.use_categories
            self.use_smoothness = opt.use_smoothness

            self.use_pose_discriminator = opt.use_pose_discriminator

            self.gan_criterion = nn.BCEWithLogitsLoss()
            self.category_criterion = nn.CrossEntropyLoss()
            self.L1_loss = nn.L1Loss()
            self.use_wgan = opt.use_wgan

            self.motion_batch_size = self.motion_sampler.batch_size
            self.pose_batch_size = self.pose_sampler.batch_size

            self.opt_generator = None
            self.opt_motion_discriminator = None
            self.opt_pose_discriminator = None

            self.real_motion_batch = None
            self.real_pose_batch = None


    def ones_like(self, t, val=1):
        return torch.Tensor(t.size()).fill_(val).requires_grad_(False).to(self.device)


    def zeros_like(self, t, val=0):
        return torch.Tensor(t.size()).fill_(val).requires_grad_(False).to(self.device)

    def calc_gradient_penalty(self, discriminator, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
        alpha = alpha.view(real_data.size())
        alpha = alpha.to(self.device)

        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.to(self.device)
        interpolates.requires_grad_(True)

        disc_interpolates, _ = discriminator(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        # gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

    def sample_real_pose_batch(self):
        if self.pose_enumerator is None:
            self.pose_enumerator = enumerate(self.pose_sampler)

        batch_idx, batch = next(self.pose_enumerator)
        if batch_idx == len(self.pose_sampler) - 1:
            self.pose_enumerator = enumerate(self.pose_sampler)
        self.real_pose_batch = batch
        return batch

    def sample_real_motion_batch(self):
        if self.motion_enumerator is None:
            self.motion_enumerator = enumerate(self.motion_sampler)

        batch_idx, batch = next(self.motion_enumerator)
        if batch_idx == len(self.motion_sampler) - 1:
            self.motion_enumerator = enumerate(self.motion_sampler)
        self.real_motion_batch = batch
        return batch

    def train_discriminator(self, discriminator, sample_true, sample_fake, optimizer, batch_size, use_categories):

        optimizer.zero_grad()

        real_batch = sample_true()
        batch = torch.clone(real_batch[0]).float().detach_().to(self.device)

        fake_batch, generated_categories = sample_fake(batch_size)

        real_labels, real_categorical = discriminator(batch)
        fake_labels, fake_categorical = discriminator(fake_batch.detach())

        ones = self.ones_like(real_labels)
        zeros = self.zeros_like(fake_labels)
        if self.use_wgan:
            l_discriminator = fake_labels.mean() - real_labels.mean()
        else:
            l_discriminator = self.gan_criterion(real_labels, ones) + \
                self.gan_criterion(fake_labels, zeros)

        if use_categories:
            categories_gt = torch.clone(real_batch[1]).long().detach_().to(self.device)
            l_category = self.category_criterion(real_categorical.squeeze(), categories_gt)
            if self.use_wgan:
                l_discriminator += l_category.mean()
            else:
                l_discriminator += l_category
            self.d_category = l_category.item()

        l_discriminator.backward()
        optimizer.step()

        if self.use_wgan:
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
        return l_discriminator

    def train_generator(self, pose_discriminator, motion_discriminator, sample_fake_poses,
                        sample_fake_motions, optimizer):

        optimizer.zero_grad()

        fake_batch, generated_categories = sample_fake_motions(self.motion_batch_size)
        fake_labels, fake_categorical = motion_discriminator(fake_batch)
        if self.use_wgan:
            l_generator = -fake_labels.mean()
        else:
            all_ones = self.ones_like(fake_labels)
            l_generator = self.gan_criterion(fake_labels, all_ones)
        l_category = self.category_criterion(fake_categorical.squeeze(), generated_categories)
        self.g_w_loss = l_generator.item()
        l_generator += 0.7 * l_category
        self.g_category = l_category.item()

        if self.use_smoothness:
            # dim (num_samples, motion_len, output_size)
            # print(fake_batch.size())
            l_smooth = self.L1_loss(fake_batch[:, :-1, :], fake_batch[:, 1:, :])
            l_generator += 0.3 * l_smooth
            self.l_smooth = l_smooth.item()

        if self.use_pose_discriminator:
            fake_batch, _ = sample_fake_poses(self.pose_batch_size)
            fake_labels, _ = pose_discriminator(fake_batch)
            if self.use_wgan:
                l_generator += 0.5 * (-1) * fake_labels.mean()
            else:
                all_ones = self.ones_like(fake_labels)
                l_generator += 0.5 * self.gan_criterion(fake_labels, all_ones)
        l_generator.backward()
        optimizer.step()
        return l_generator

    @staticmethod
    def evaluate(generator, z_m, z_c, num_samples):
        generator.eval()
        with torch.no_grad():
            z = generator.create_z_motion(z_m, z_c, num_samples)
            fake, _ = generator.generate_motion_fixed_noise(z, num_samples)
            fake = fake.permute(1, 0, 2)
            return fake.detach().cpu()

    def fake_pose_preprocess_dis(self, fake, num_samples, real_pose_batch):
        pass

    def fake_motion_preprocess_dis(self, fake, num_samples, real_joints):
        pass

    def train(self, generator, pose_discriminator, motion_discriminator):

        generator.to(self.device)
        pose_discriminator.to(self.device)
        motion_discriminator.to(self.device)
        if self.use_wgan:
            self.opt_generator = optim.RMSprop(generator.parameters(), lr=0.00005)

            if self.use_pose_discriminator:
                self.opt_pose_discriminator = optim.RMSprop(pose_discriminator.parameters(), lr=0.00005)

            self.opt_motion_discriminator = optim.RMSprop(motion_discriminator.parameters(), lr=0.00005)
        else:
            self.opt_generator = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

            if self.use_pose_discriminator:
                self.opt_pose_discriminator = optim.Adam(pose_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

            self.opt_motion_discriminator = optim.Adam(motion_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

        def sample_fake_pose_batch(batch_size):
            return generator.sample_poses(batch_size)

        def sample_fake_motion_batch(batch_size):
            return generator.sample_motion_clips(batch_size)

        def init_logs():
            log_dict = {'l_gen': [0], 'l_pose_dis': [0], 'l_motion_dis': [0]}
            if self.use_smoothness:
                log_dict['g_smooth'] = [0]
            log_dict['g_category'] = [0]
            log_dict['d_category'] = [0]
            return log_dict

        batch_num = 0

        logs = init_logs()

        start_time = time.time()

        e_num_samples = 20
        fixed_m_noise = generator.sample_z_r(e_num_samples)
        fixed_c_noise, classes = generator.sample_z_categ(e_num_samples)

        np.save(os.path.join(self.opt.joints_path, "motion_class.npy"), classes)

        while True:
            generator.train()
            pose_discriminator.train()
            motion_discriminator.train()

            self.opt_generator.zero_grad()
            self.opt_motion_discriminator.zero_grad()
            l_pose_dis = 0

            if self.use_pose_discriminator:
                self.opt_pose_discriminator.zero_grad()
                l_pose_dis = self.train_discriminator(pose_discriminator, self.sample_real_pose_batch,
                                                      sample_fake_pose_batch, self.opt_pose_discriminator,
                                                      self.pose_batch_size, use_categories=False)

            l_motion_dis = self.train_discriminator(motion_discriminator, self.sample_real_motion_batch,
                                                    sample_fake_motion_batch, self.opt_motion_discriminator,
                                                    self.motion_batch_size, use_categories=self.use_categories)

            l_gen = self.train_generator(pose_discriminator, motion_discriminator,
                                         sample_fake_pose_batch, sample_fake_motion_batch,
                                         self.opt_generator)
            logs['l_gen'].append(l_gen.item())
            logs['l_pose_dis'].append(l_pose_dis.item())
            logs['l_motion_dis'].append(l_motion_dis.item())
            logs['d_category'].append(self.d_category)
            logs['g_category'].append(self.g_category)
            if self.use_smoothness:
                logs['g_smooth'].append(self.l_smooth)
            batch_num += 1

            if batch_num % self.opt.print_every == 0:
                mean_loss = init_logs()
                for k, v in logs.items():
                    mean_loss[k] = sum(logs[k][-1 * self.opt.print_every:]) / self.opt.print_every
                print_current_loss(start_time, batch_num, self.opt.iterations, mean_loss)

            if batch_num % self.opt.eval_every == 0:
                fake_motion = self.evaluate(generator, fixed_m_noise, fixed_c_noise, e_num_samples).numpy()
                np.save(os.path.join(self.opt.joints_path, "motion_joints" + str(batch_num) + ".npy"), fake_motion)

            if batch_num % self.opt.save_every == 0:
                state = {
                    "generator": generator.state_dict(),
                    "pose_discriminator": pose_discriminator.state_dict(),
                    "motion_discriminator": motion_discriminator.state_dict(),
                    "opt_generator": self.opt_generator.state_dict(),
                    "opt_pose_discriminator": self.opt_pose_discriminator.state_dict(),
                    "opt_motion_discriminator": self.opt_motion_discriminator.state_dict(),
                    "epoch": batch_num
                }
                torch.save(state, os.path.join(self.opt.model_path, str(batch_num) + ".tar"))

            if batch_num % self.opt.save_latest == 0:
                state = {
                    "generator": generator.state_dict(),
                    "pose_discriminator": pose_discriminator.state_dict(),
                    "motion_discriminator": motion_discriminator.state_dict(),
                    "opt_generator": self.opt_generator.state_dict(),
                    "opt_pose_discriminator": self.opt_pose_discriminator.state_dict(),
                    "opt_motion_discriminator": self.opt_motion_discriminator.state_dict(),
                    "epoch": batch_num
                }
                torch.save(state, os.path.join(self.opt.model_path, "latest.tar"))

            if batch_num >= self.opt.iterations:
                break
        return logs


class TrainerLie(Trainer):
    def __init__(self, pose_sampler, motion_sampler, opt, device, raw_offsets, kinematic_chain):
        super(TrainerLie, self).__init__(pose_sampler,
                                        motion_sampler,
                                        opt,
                                        device)
        self.raw_offsets = torch.from_numpy(raw_offsets).to(device).detach()
        self.kinematic_chain = kinematic_chain
        self.Tensor = torch.cuda.FloatTensor if self.opt.gpu_id != '' else torch.Tensor
        self.lie_skeleton = LieSkeleton(self.raw_offsets, kinematic_chain, self.Tensor)

    def fake_pose_preprocess_dis(self, fake, num_samples, real_pose_batch=None):
        if self.opt.no_trajectory:
            fake_lie_params = fake
            root_translation = self.zeros_like(fake[..., -3:], 0)
        else:
            fake_lie_params = fake[..., : -3]
            root_translation = fake[..., -3:]
        # for pose, there is no translation at root joints
        zero_root_translation = self.zeros_like(root_translation, 0)
        zero_padding = self.zeros_like(root_translation, 0)
        fake_lie_params = torch.cat((zero_padding, fake_lie_params), dim=-1)
        if real_pose_batch is None:
            real_pose_batch, _ = self.sample_real_pose_batch()
        real_pose_batch = real_pose_batch.to(self.device)
        real_pose_batch = real_pose_batch.view(num_samples, -1, 3)
        fake_pose_joints = self.lie_to_joints(fake_lie_params, real_pose_batch, zero_root_translation)
        return fake_pose_joints

    def fake_motion_preprocess_dis(self, fake, num_samples, real_joints):
        if self.opt.no_trajectory:
            fake_lie_params = fake
            root_translation = self.zeros_like(fake[..., -3:], 0)
        else:
            fake_lie_params = fake[..., : -3]
            root_translation = fake[..., -3:]
        zero_padding = self.zeros_like(root_translation, 0)
        fake_lie_params = torch.cat((zero_padding, fake_lie_params), dim=-1)
        motion_len = fake.shape[1]
        # real joints for get the bone length in real scenario
        if real_joints is None:
            real_motion_batch, _ = self.sample_real_motion_batch()
            real_joints = self.real_motion_batch[:, 0, :]
        real_joints = real_joints.to(self.device)
        real_joints = real_joints.view(num_samples, -1, 3)
        fake_steps = []
        for i in range(motion_len):
            # (batch_size, joints * 3)
            step_i_poses = self.lie_to_joints(fake_lie_params[:, i, :], real_joints, root_translation[:, i, :])
            step_i_poses = step_i_poses.unsqueeze(1)
            fake_steps.append(step_i_poses)
        fake_motion = torch.cat(fake_steps, dim=1)
        return fake_motion

    def train(self, generator, pose_discriminator, motion_discriminator):

        generator.to(self.device)
        pose_discriminator.to(self.device)
        motion_discriminator.to(self.device)

        if self.use_wgan:
            self.opt_generator = optim.RMSprop(generator.parameters(), lr=0.00005)

            if self.use_pose_discriminator:
                self.opt_pose_discriminator = optim.RMSprop(pose_discriminator.parameters(), lr=0.00005)

            self.opt_motion_discriminator = optim.RMSprop(motion_discriminator.parameters(), lr=0.00005)
        else:
            self.opt_generator = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

            if self.use_pose_discriminator:
                self.opt_pose_discriminator = optim.Adam(pose_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

            self.opt_motion_discriminator = optim.Adam(motion_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

        if self.opt.is_continue:
            model = torch.load(os.path.join(self.opt.model_path, 'latest.tar'))
            generator.load_state_dict(model['generator'])
            motion_discriminator.load_state_dict(model['motion_discriminator'])
            self.opt_generator.load_state_dict(model['opt_generator'])
            self.opt_motion_discriminator.load_state_dict(model['opt_motion_discriminator'])

            if self.use_pose_discriminator:
                pose_discriminator.load_state_dict(model['pose_discriminator'])
                self.opt_pose_discriminator.load_state_dict(model['opt_pose_discriminator'])

        def sample_fake_pose_batch(batch_size):
            # (batch_size, joints_num * 3)
            fake, _ = generator.sample_poses(batch_size)
            fake_pose_joints = self.fake_pose_preprocess_dis(fake, batch_size, self.real_pose_batch[0])
            return fake_pose_joints, None

        def sample_fake_motion_batch(batch_size):
            # (batch_size, motion_len, joints_num * 3)
            fake, categories = generator.sample_motion_clips(batch_size)
            real_joints = self.real_motion_batch[0][:, 0, :]
            fake_motion = self.fake_motion_preprocess_dis(fake, batch_size, real_joints)
            return fake_motion, categories

        def init_logs():
            log_dict = {'l_gen': [0], 'l_pose_dis': [0], 'l_motion_dis': [0]}
            if self.use_smoothness:
                log_dict['g_smooth'] = [0]
            log_dict['g_category'] = [0]
            log_dict['d_category'] = [0]
            return log_dict

        batch_num = 0

        logs = init_logs()

        start_time = time.time()

        e_num_samples = 20
        fixed_m_noise = generator.sample_z_r(e_num_samples)
        fixed_c_noise, classes = generator.sample_z_categ(e_num_samples)

        np.save(os.path.join(self.opt.joints_path, "motion_class.npy"), classes)

        while True:
            generator.train()
            pose_discriminator.train()
            motion_discriminator.train()

            self.opt_generator.zero_grad()
            self.opt_motion_discriminator.zero_grad()
            l_pose_dis = 0

            if self.use_pose_discriminator:
                self.opt_pose_discriminator.zero_grad()
                l_pose_dis = self.train_discriminator(pose_discriminator, self.sample_real_pose_batch,
                                                      sample_fake_pose_batch, self.opt_pose_discriminator,
                                                      self.pose_batch_size, use_categories=False)

            l_motion_dis = self.train_discriminator(motion_discriminator, self.sample_real_motion_batch,
                                                    sample_fake_motion_batch, self.opt_motion_discriminator,
                                                    self.motion_batch_size, use_categories=self.use_categories)

            l_gen = self.train_generator(pose_discriminator, motion_discriminator,
                                         sample_fake_pose_batch, sample_fake_motion_batch,
                                         self.opt_generator)
            logs['l_gen'].append(l_gen.item())
            logs['l_pose_dis'].append(l_pose_dis.item())
            logs['l_motion_dis'].append(l_motion_dis.item())
            logs['d_category'].append(self.d_category)
            logs['g_category'].append(self.g_category)
            if self.use_smoothness:
                logs['g_smooth'].append(self.l_smooth)
            batch_num += 1

            if batch_num % self.opt.print_every == 0:
                mean_loss = init_logs()
                for k, v in logs.items():
                    mean_loss[k] = sum(logs[k][-1 * self.opt.print_every:]) / self.opt.print_every
                print_current_loss(start_time, batch_num, self.opt.iterations, mean_loss)

            if batch_num % self.opt.eval_every == 0:
                fake_motion = self.evaluate(generator, fixed_m_noise, fixed_c_noise, e_num_samples).numpy()
                np.save(os.path.join(self.opt.joints_path, "motion_joints" + str(batch_num) + ".npy"), fake_motion)

            if batch_num % self.opt.save_every == 0:
                state = {
                    "generator": generator.state_dict(),
                    "pose_discriminator": pose_discriminator.state_dict(),
                    "motion_discriminator": motion_discriminator.state_dict(),
                    "opt_generator": self.opt_generator.state_dict(),
                    "opt_pose_discriminator": self.opt_pose_discriminator.state_dict(),
                    "opt_motion_discriminator": self.opt_motion_discriminator.state_dict(),
                    "epoch": batch_num
                }
                torch.save(state, os.path.join(self.opt.model_path, str(batch_num) + ".tar"))

            if batch_num % self.opt.save_latest == 0:
                state = {
                    "generator": generator.state_dict(),
                    "pose_discriminator": pose_discriminator.state_dict(),
                    "motion_discriminator": motion_discriminator.state_dict(),
                    "opt_generator": self.opt_generator.state_dict(),
                    "opt_pose_discriminator": self.opt_pose_discriminator.state_dict(),
                    "opt_motion_discriminator": self.opt_motion_discriminator.state_dict(),
                    "epoch": batch_num
                }
                torch.save(state, os.path.join(self.opt.model_path, "latest.tar"))

            if batch_num >= self.opt.iterations:
                break
        return logs

    # convert the lie params of a batch of pose to 3d coordinates
    def lie_to_joints(self, lie_params, joints, root_translation):
        lie_params = lie_params.view(lie_params.shape[0], -1, 3)
        joints = self.lie_skeleton.forward_kinematics(lie_params, joints, root_translation)
        return joints.view(joints.shape[0], -1)

    def evaluate(self, generator, z_m, z_c, num_samples):
        generator.eval()
        with torch.no_grad():
            z = generator.create_z_motion(z_m, z_c, num_samples)
            fake, _ = generator.generate_motion_fixed_noise(z, num_samples)
            fake = fake.permute(1, 0, 2)
            if self.opt.isTrain:
                real_joints = self.real_pose_batch
                real_joints = real_joints[0].to(self.device)
            else:
                real_joints = self.sample_real_pose_batch()
                real_joints = real_joints[0].to(self.device)

            if real_joints.shape[0] < num_samples:
                repeat_ratio = int(num_samples / real_joints.shape[0])
                real_joints = real_joints.repeat((repeat_ratio, 1))
                pad_num = num_samples - real_joints.shape[0]
                if pad_num != 0:
                    real_joints = torch.cat((real_joints, real_joints[: pad_num]), dim=0)
            else:
                real_joints = real_joints[:num_samples]
            fake_motion = self.fake_motion_preprocess_dis(fake, num_samples, real_joints)
            return fake_motion.detach().cpu()


class TrainerLieV2(TrainerLie):
    def __init__(self, pose_sampler, motion_sampler, opt, device, raw_offsets, kinematic_chain):
        super(TrainerLieV2, self).__init__(pose_sampler,
                                           motion_sampler,
                                           opt,
                                           device,
                                           raw_offsets,
                                           kinematic_chain)

    def fake_motion_preprocess_dis(self, fake, num_samples, real_joints=None):
        return fake

    def fake_pose_preprocess_dis(self, fake, num_samples, real_pose_batch=None):
        if self.opt.no_trajectory:
            return fake
        else:
            fake[..., -3:] = 0
            return fake

    def evaluate(self, generator, z_m, z_c, num_samples):
        generator.eval()
        with torch.no_grad():
            z = generator.create_z_motion(z_m, z_c, num_samples)
            fake, _ = generator.generate_motion_fixed_noise(z, num_samples)
            fake = fake.permute(1, 0, 2)
            fake_motion = fake
            if not self.opt.isTrain:
                real_joints = self.sample_real_pose_batch()
                real_joints = real_joints[0].to(self.device)

                if real_joints.shape[0] < num_samples:
                    repeat_ratio = int(num_samples / real_joints.shape[0])
                    real_joints = real_joints.repeat((repeat_ratio, 1))
                    pad_num = num_samples - real_joints.shape[0]
                    if pad_num != 0:
                        real_joints = torch.cat((real_joints, real_joints[: pad_num]), dim=0)
                else:
                    real_joints = real_joints[:num_samples]

                fake_motion = super(TrainerLieV2, self).fake_motion_preprocess_dis(fake, num_samples, real_joints)
            return fake_motion.detach().cpu()


class TrainerV2(Trainer):
    def __init__(self, pose_sampler, motion_sampler, opt, device):
        super(TrainerV2, self).__init__(pose_sampler,
                                        motion_sampler,
                                        opt,
                                        device)

    def train_discriminator_v2(self, discriminator, sample_true, sample_fake, optimizer, batch_size, use_categories,
                               use_wgan):
        optimizer.zero_grad()

        real_batch = sample_true()
        batch = torch.clone(real_batch[0]).float().detach_().to(self.device)

        fake_batch, generated_categories = sample_fake(batch_size)
        real_labels, _ = discriminator(batch)
        fake_labels, _ = discriminator(fake_batch.detach())

        if use_wgan:
            l_discriminator = fake_labels.mean() - real_labels.mean()

        if use_categories:
            categories_gt = torch.clone(real_batch[1]).long().detach_().to(self.device)
            l_category = self.category_criterion(real_labels.squeeze(), categories_gt)
            l_discriminator = l_category
            self.d_category = l_category.item()

        l_discriminator.backward()
        optimizer.step()
        if use_wgan:
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
        return l_discriminator

    def train_generator_v2(self, pose_discriminator, motion_discriminator, motion_classifier, sample_fake_poses,
                        sample_fake_motions, optimizer):
        optimizer.zero_grad()

        fake_batch, generated_categories = sample_fake_motions(self.motion_batch_size)
        fake_labels, _ = motion_discriminator(fake_batch)
        if self.use_wgan:
            l_generator = -fake_labels.mean()
        else:
            all_ones = self.ones_like(fake_labels)
            l_generator = self.gan_criterion(fake_labels, all_ones)
        fake_categorical, _ = motion_classifier(fake_batch)
        l_category = self.category_criterion(fake_categorical.squeeze(), generated_categories)
        self.g_w_loss = l_generator.item()
        l_generator += 0.7 * l_category
        self.g_category = l_category.item()

        if self.use_smoothness:
            # dim (num_samples, motion_len, output_size)
            # print(fake_batch.size())
            l_smooth = self.L1_loss(fake_batch[:, :-1, :], fake_batch[:, 1:, :])
            l_generator += 0.3 * l_smooth
            self.l_smooth = l_smooth.item()

        if self.use_pose_discriminator:
            fake_batch, _ = sample_fake_poses(self.pose_batch_size)
            fake_labels, _ = pose_discriminator(fake_batch)
            if self.use_wgan:
                l_generator += 0.5 * (-1) * fake_labels.mean()
            else:
                all_ones = self.ones_like(fake_labels)
                l_generator += 0.5 * self.gan_criterion(fake_labels, all_ones)
        l_generator.backward()
        optimizer.step()
        return l_generator

    def train_v2(self, generator, pose_discriminator, motion_discriminator, motion_classifier):

        generator.to(self.device)
        pose_discriminator.to(self.device)
        motion_discriminator.to(self.device)
        motion_classifier.to(self.device)
        if self.use_wgan:
            self.opt_generator = optim.RMSprop(generator.parameters(), lr=0.00005)

            if self.use_pose_discriminator:
                self.opt_pose_discriminator = optim.RMSprop(pose_discriminator.parameters(), lr=0.00005)

            self.opt_motion_discriminator = optim.RMSprop(motion_discriminator.parameters(), lr=0.00005)
        else:
            self.opt_generator = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

            if self.use_pose_discriminator:
                self.opt_pose_discriminator = optim.Adam(pose_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

            self.opt_motion_discriminator = optim.Adam(motion_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

        self.opt_classifier = optim.Adam(motion_classifier.parameters(), lr=0.0002, betas=(0.5, 0.999),
                                                   weight_decay=0.00001)

        def sample_fake_pose_batch(batch_size):
            return generator.sample_poses(batch_size)

        def sample_fake_motion_batch(batch_size):
            return generator.sample_motion_clips(batch_size)

        def init_logs():
            log_dict = {'l_gen': [0], 'l_pose_dis': [0], 'l_motion_dis': [0]}
            if self.use_smoothness:
                log_dict['g_smooth'] = [0]
            log_dict['g_category'] = [0]
            log_dict['l_motion_cls'] = [0]
            log_dict['g_w_loss'] = [0]
            return log_dict

        batch_num = 0

        logs = init_logs()

        start_time = time.time()

        e_num_samples = 20
        fixed_m_noise = generator.sample_z_r(e_num_samples)
        fixed_c_noise, classes = generator.sample_z_categ(e_num_samples)
        np.save(os.path.join(self.opt.joints_path, "motion_class.npy"), classes)

        while True:
            generator.train()
            pose_discriminator.train()
            motion_discriminator.train()
            motion_classifier.train()

            self.opt_generator.zero_grad()
            self.opt_motion_discriminator.zero_grad()
            self.opt_pose_discriminator.zero_grad()
            self.opt_classifier.zero_grad()

            if self.use_pose_discriminator:
                self.opt_pose_discriminator.zero_grad()
                l_pose_dis = self.train_discriminator_v2(pose_discriminator, self.sample_real_pose_batch,
                                                      sample_fake_pose_batch, self.opt_pose_discriminator,
                                                      self.pose_batch_size, use_categories=False, use_wgan=True)

            l_motion_dis = self.train_discriminator_v2(motion_discriminator, self.sample_real_motion_batch,
                                                    sample_fake_motion_batch, self.opt_motion_discriminator,
                                                    self.motion_batch_size, use_categories=False, use_wgan=True)

            l_motion_cls = self.train_discriminator_v2(motion_classifier, self.sample_real_motion_batch,
                                                    sample_fake_motion_batch, self.opt_classifier,
                                                    self.motion_batch_size, use_categories=True, use_wgan=False)

            l_gen = self.train_generator_v2(pose_discriminator, motion_discriminator, motion_classifier,
                                         sample_fake_pose_batch, sample_fake_motion_batch,
                                         self.opt_generator)
            logs['l_gen'].append(l_gen.item())
            logs['l_pose_dis'].append(l_pose_dis.item())
            logs['l_motion_dis'].append(l_motion_dis.item())
            logs['l_motion_cls'].append(l_motion_cls.item())
            logs['g_category'].append(self.g_category)
            logs['g_w_loss'].append(self.g_w_loss)
            if self.use_smoothness:
                logs['g_smooth'].append(self.l_smooth)
            batch_num += 1

            if batch_num % self.opt.print_every == 0:
                mean_loss = init_logs()
                for k, v in logs.items():
                    mean_loss[k] = sum(logs[k][-1 * self.opt.print_every:]) / self.opt.print_every
                print_current_loss(start_time, batch_num, self.opt.iterations, mean_loss)

            if batch_num % self.opt.eval_every == 0:
                fake_motion = self.evaluate(generator, fixed_m_noise, fixed_c_noise, e_num_samples).numpy()
                np.save(os.path.join(self.opt.joints_path, "motion_joints" + str(batch_num) + ".npy"), fake_motion)

                state = {
                    "generator": generator.state_dict(),
                    "pose_discriminator": pose_discriminator.state_dict(),
                    "motion_discriminator": motion_discriminator.state_dict(),
                    "opt_generator": self.opt_generator.state_dict(),
                    "opt_pose_discriminator": self.opt_pose_discriminator.state_dict(),
                    "opt_motion_discriminator": self.opt_motion_discriminator.state_dict(),
                    "epoch": batch_num
                }
                torch.save(state, os.path.join(self.opt.model_path, str(batch_num) + ".tar"))
            if batch_num >= self.opt.iterations:
                break
        return logs
