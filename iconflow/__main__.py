# Licensed under CC BY-NC-SA 4.0

import os
import warnings

try:
    import ei
    ei.patch()
except ImportError:
    pass


class Main:
    
    def __init__(
        self,
        output_dir = './output',
        dataset_dir = './dataset',
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        assert os.path.isdir(dataset_dir)
        self.dataset_dir = dataset_dir
        
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        warnings.filterwarnings("ignore")
    

    def train_net(
        self,
        
        device = 'cpu',
        batch_size = 32,
        num_workers = 8,
        train_ratio = 0.9,
        
        log_interval = 100,
        sample_interval = 1000,
        save_interval = 500,
        save_iterations = [100000, 200000, 300000, 400000, 500000],
        end_iteration = 600000,
    ):
        import os
        import random
        import torch
        import torch.optim as optim
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from torch.utils.tensorboard.writer import SummaryWriter
        from torchvision.utils import make_grid
        import einops
        from tqdm import tqdm
        from .dataset import IconContourDataset
        from .utils.train import infinite_loop, random_sampler
        from .model.colorizer import ReferenceBasedColorizer
        
        image_size = 128
        device = torch.device(device)
        writer = SummaryWriter(self.output_dir)
        
        # Datasets
        data_dir = os.path.join(self.dataset_dir, 'data')
        train_set = IconContourDataset(data_dir, image_size, split=(0, train_ratio))
        test_set = IconContourDataset(data_dir, image_size, split=(train_ratio, 1))
        net_train_set = IconContourDataset(
            data_dir, image_size,
            random_crop=True,
            random_transpose=True,
            random_color=True,
            split=(0, train_ratio),
            normalize=True,
        )
        net_train_loader = DataLoader(
            net_train_set,
            batch_size=batch_size,
            sampler=random_sampler(len(net_train_set)),
            pin_memory=(device.type == 'cuda'),
            num_workers=num_workers
        )
        
        net = ReferenceBasedColorizer()
        opt = optim.Adam(net.parameters(), lr=1e-4)
        net.to(device)
        
        # Try to load training state from checkpoint
        try:
            state_path = os.path.join(self.output_dir, 'checkpoint.pt')
            state = torch.load(state_path, map_location=device)
            net.load_state_dict(state['net'])
            opt.load_state_dict(state['opt'])
            it = state['it']
            print(f'net loaded from {state_path}, iteration: {it}')
            del state
        except FileNotFoundError:
            it = 0
        
        @torch.no_grad()
        def display_rec(k=4):
            rand = random.Random(it % 300000 + 1337)
            image, contour = zip(
                *rand.choices(train_set, k=k),
                *rand.choices(test_set, k=k)
            )
            image, contour = map(torch.stack, [image, contour])
            
            net.eval()
            reconstructions = []
            for i in range(k + k):
                rolled_image = torch.roll(image, i, 0) # roll on batch dim
                reconstruction = net(contour.to(device), rolled_image.to(device)).cpu()
                reconstructions.append(reconstruction)
            
            display_columns = [
                image,
                einops.repeat(contour, 'B 1 H W -> B 3 H W'),
                *reconstructions
            ]
            display_image = (torch.stack(display_columns, 1) + 0.5).clamp(0, 1)
            display_image = einops.rearrange(display_image, 'row col ... -> (row col) ...')
            display_image = make_grid(display_image, len(display_columns))
            return display_image

        @torch.no_grad()
        def display_ext(k=4):
            rand = random.Random(it % 300000 + 1337)
            image, contour = zip(
                *rand.choices(train_set, k=k),
                *rand.choices(test_set, k=k)
            )
            image, contour = map(torch.stack, (image, contour))
            
            net.eval()
            ext = net.extract_content(image.to(device)).cpu()
            rec = net(contour.to(device), image.roll(1, 0).to(device))
            
            rec_ext = net.extract_content(rec).cpu()
            rec = rec.cpu()
            
            display_columns = [
                einops.repeat(contour, 'B 1 H W -> B 3 H W'),
                image,
                einops.repeat(ext, 'B 1 H W -> B 3 H W'),
                rec,
                einops.repeat(rec_ext, 'B 1 H W -> B 3 H W')
            ]
            display_image = (torch.stack(display_columns, 1) + 0.5).clamp(0, 1)
            display_image = einops.rearrange(display_image, 'row col ... -> (row col) ...')
            display_image = make_grid(display_image, len(display_columns))
            return display_image
        
        def save_checkpoint(suffix=''):
            name = f'checkpoint{suffix}.pt'
            torch.save({
                'net': net.state_dict(),
                'opt': opt.state_dict(),
                'it': it
            }, os.path.join(self.output_dir, name))
        
        def train_step(image, contour):
            image1 = image.to(device)
            contour1 = contour.to(device)
            
            net.train()
            ext1 = net.encode_content(contour1)
            style1 = net.encode_style(image1)
            style2 = style1.roll(1, 0).contiguous()
            rec11 = net.decode(ext1, style1)
            rec12 = net.decode(ext1, style2)
            
            log_dict = {}
            loss = 0
            
            # Reconstruction Error
            reconstruction_error = F.mse_loss(rec11, image1)
            loss = loss + reconstruction_error
            log_dict['RE'] = reconstruction_error.item()
            
            # Extraction Error
            extraction_error = torch.stack([
                F.mse_loss(net.extract_content(image1), contour1),
            ]).mean()
            loss = loss + extraction_error
            log_dict['EE'] = extraction_error.item()
        
            # Content Consistency
            net.content_extractor.requires_grad_(False)
            content_consistency = torch.stack([
                F.mse_loss(net.extract_content(rec12), contour1)
            ]).mean()
            net.content_extractor.requires_grad_(True)
            loss = loss + content_consistency
            log_dict['CC'] = content_consistency.item()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            return log_dict

        try:
            with tqdm(infinite_loop(net_train_loader), total=end_iteration-it) as loader:
                for image, contour in loader:
                    if isinstance(end_iteration, int) and it >= end_iteration:
                        raise KeyboardInterrupt
                    
                    log_dict = train_step(image, contour)
                    loader.set_postfix({'it': it, **log_dict})
    
                    if it % log_interval == 0:
                        for key, value in log_dict.items():
                            writer.add_scalar(f'train/{key}', value, it)
                    
                    if (it < 1000 and it % 200 == 0) or (it >= 1000 and it % sample_interval == 0):
                        writer.add_image(f'sample/reconstruct', display_rec(), it)
                        writer.add_image(f'sample/extract', display_ext(), it)
                    
                    it += 1
                    
                    if it % save_interval == 0:
                        save_checkpoint()
                        
                    if it in save_iterations:
                        save_checkpoint(suffix=f'_{it}')

        except KeyboardInterrupt:
            print('saving checkpoint')
            save_checkpoint()
    
    
    def train_flow(
        self,
        
        device = 'cpu',
        batch_size = 64,
        num_workers = 8,
        train_ratio = 0.9,
        max_samples = 1000,
        
        log_interval = 50,
        sample_interval = 200,
        save_interval = 200,
        save_iterations = [(i+1) * 10000 for i in range(30)],
        end_iteration = 10000 * 30,
    ):
        import os
        import math
        import random
        import torch
        import numpy as np
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from torch.utils.tensorboard.writer import SummaryWriter
        from torch.distributions import Normal
        from torchvision.utils import make_grid
        from tqdm import tqdm

        from .utils.train import infinite_loop, random_sampler
        from .utils.style import get_style_image
        from .utils.image import from_image1
        from .model import ReferenceBasedColorizer, get_flow
        from .dataset import StylePaletteDataset, IconContourDataset
        
        flow_output_dir = os.path.join(self.output_dir, 'flow')
        os.makedirs(flow_output_dir, exist_ok=True)
        
        image_size = 128
        device = torch.device(device)
        writer = SummaryWriter(flow_output_dir)

        # Datasets
        data_dir = os.path.join(self.dataset_dir, 'data')
        train_set = IconContourDataset(data_dir, image_size, split=(0, train_ratio))
        test_set = IconContourDataset(data_dir, image_size, split=(train_ratio, 1))
        flow_train_set = StylePaletteDataset(data_dir, image_size, self.dataset_dir, max_samples, num_workers=num_workers)
        flow_train_loader = DataLoader(
            flow_train_set,
            batch_size=batch_size,
            sampler=random_sampler(len(flow_train_set)),
            pin_memory=(device.type == 'cuda'),
            num_workers=num_workers
        )
        
        # Reference-Based Colorizer
        net = ReferenceBasedColorizer()
        net.to(device)
        net_state_path = os.path.join(self.output_dir, 'checkpoint.pt')
        net_state = torch.load(net_state_path, map_location=device)
        net.load_state_dict(net_state['net'])
        print('net loaded from {}, iteration: {}'.format(net_state_path, net_state['it']))
        del net_state

        # Normalizing Flow
        flow = get_flow(net.style_dim)
        opt = optim.Adam(flow.parameters(), lr=1e-3)
        flow.to(device)
        
        # Try to load flow training state from checkpoint
        try:
            state_path = os.path.join(flow_output_dir, 'checkpoint.pt')
            state = torch.load(state_path, map_location=device)
            flow.load_state_dict(state['flow'])
            opt.load_state_dict(state['opt'])
            it = state['it']
            print(f'flow loaded from {state_path}, iteration: {it}')
            del state
        except FileNotFoundError:
            it = 0
        
        base_distribution = Normal(0, 1)
        
        def display_sample(k=4):
            temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            
            condition_train = random.Random(it).choice(train_set)[1]
            condition_test = random.Random(it).choice(test_set)[1]
            condition_batch = [condition_train] * k + [condition_test] * k
            condition = torch.stack(condition_batch).to(device)

            selected_style_names = np.random.choice(flow_train_set.style_names, size=(k + k))
            
            selected_style_images = torch.stack([
                from_image1(get_style_image(flow_train_set.style_to_cmb[name], name))
                for name in selected_style_names
            ])
            
            location = torch.stack([
                flow_train_set.position_to_condition(flow_train_set.style_to_pos[name])
                for name in selected_style_names
            ]).to(device)
            
            net.eval()
            flow.eval()
            display_columns = [selected_style_images]
            for t in temperatures:
                with torch.no_grad():
                    z = base_distribution.sample([condition.shape[0], net.style_dim]) * t
                    z = z.to(device)
                    embed = net.encode_content(condition)
                    style = flow(z, location, reverse=True)
                    reconstruction = net.decode(embed, style).cpu()
                    display_columns.append(reconstruction)
                    del z, embed, style, reconstruction
            
            display_image = torch.clamp(torch.cat(display_columns) + 0.5, 0, 1)
            display_image = make_grid(display_image, nrow=(len(temperatures) + 1))
            return display_image
        
        def save_checkpoint(suffix=''):
            name = f'checkpoint{suffix}.pt'
            torch.save({
                'flow': flow.state_dict(),
                'opt': opt.state_dict(),
                'it': it,
            }, os.path.join(flow_output_dir, name))
        
        def train_step(image, location):
            image = image.to(device)
            location = location.to(device)
            
            net.eval()
            with torch.no_grad():
                style = net.encode_style(image)
                style = style.reshape(batch_size, -1)

            flow.train()
            zero = torch.zeros(batch_size, 1, device=device)
            z, dlogp = flow(style, location, zero)

            logpz = Normal(0, 1).log_prob(z).sum(-1)
            logpx = logpz - dlogp
            nll = -logpx.mean()
            bpd = nll / net.style_dim / math.log(2)
            
            opt.zero_grad()
            bpd.backward()
            opt.step()
            
            return bpd.item()

        try:
            with tqdm(infinite_loop(flow_train_loader), total=end_iteration-it) as bar:
                for images, locations in bar:
                    if it >= end_iteration:
                        raise KeyboardInterrupt
                        
                    loss = train_step(images, locations)
                    bar.set_postfix({'loss': loss})

                    if it % log_interval == 0:
                        writer.add_scalar('train/loss', loss, it)
                    
                    if it % sample_interval == 0:
                        writer.add_image(f'sample/sample', display_sample(), it)
                                        
                    it += 1
                    
                    if it % save_interval == 0:
                        save_checkpoint()
                    
                    if it in save_iterations:
                        save_checkpoint(suffix=f'_{it}')


        except KeyboardInterrupt:
            print('saving checkpoint')
            save_checkpoint()


    def train_up(
        self,
        
        image_size,
        device,
        batch_size = 8,
        num_workers = 12,
        train_ratio = 0.9,
        
        log_interval = 100,
        sample_interval = 1000,
        save_interval = 500,
        save_iterations = [(i+1) * 10000 for i in range(30)],
        end_iteration = 10000 * 30,
    ):
        import os
        import random
        import einops
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from torch.utils.data.dataloader import DataLoader
        from torch.utils.tensorboard.writer import SummaryWriter
        from torchvision.utils import make_grid
        from tqdm import tqdm
        from .model import ReferenceBasedColorizer, Upsampler128to256, Upsampler128to512
        from .dataset import IconContourDownscaleDataset
        from .utils.train import infinite_loop, random_sampler
        
        up_output_dir = os.path.join(self.output_dir, f'up_{image_size}')
        os.makedirs(up_output_dir, exist_ok=True)
        
        assert image_size in (256, 512)
        device = torch.device(device)
        writer = SummaryWriter(up_output_dir)
        
        # Datasets
        data_dir = os.path.join(self.dataset_dir, 'data')
        train_set = IconContourDownscaleDataset(data_dir, image_size, 128, split=(0, train_ratio))
        test_set = IconContourDownscaleDataset(data_dir, image_size, 128, split=(train_ratio, 1))
        up_train_set = IconContourDownscaleDataset(data_dir, image_size, 128, True, True, True, (0, train_ratio))
        up_train_loader = DataLoader(
            up_train_set,
            batch_size=batch_size,
            sampler=random_sampler(len(up_train_set)),
            pin_memory=(device.type == 'cuda'),
            num_workers=num_workers
        )
        
        # Reference-Based Colorizer
        net = ReferenceBasedColorizer()
        net.to(device)
        net_state_path = os.path.join(self.output_dir, 'checkpoint.pt')
        net_state = torch.load(net_state_path, map_location=device)
        net.load_state_dict(net_state['net'])
        print('net loaded from {}, iteration: {}'.format(net_state_path, net_state['it']))
        del net_state
        
        # Upsampler
        up: nn.Module = {
            256: Upsampler128to256,
            512: Upsampler128to512
        }[image_size]()
        opt = optim.Adam(up.parameters(), lr=1e-4)
        up.to(device)
        # Try to load upsampler training state from checkpoint
        try:
            up_state_path = os.path.join(up_output_dir, 'checkpoint.pt')
            up_state = torch.load(up_state_path, map_location=device)
            up.load_state_dict(up_state['up'])
            opt.load_state_dict(up_state['opt'])
            it = up_state['it']
            print(f'up loaded from {up_state_path}, iteration: {it}')
            del up_state
        except:
            it = 0
        
        @torch.no_grad()
        def display_sample():
            k = {256: 2, 512: 1}[image_size]
            rand = random.Random(it % 300000 + 1337)
            image, contour, _, contour_hr = zip(
                *rand.choices(train_set, k=k),
                *rand.choices(test_set, k=k)
            )
            image, contour, contour_hr = map(torch.stack, [image, contour, contour_hr])
            
            up.eval()
            rec_list = []
            rec_hr_list = []
            for i in range(k + k):
                rolled_image = torch.roll(image, i, 0)
                rec = net(contour.to(device), rolled_image.to(device))
                rec_hr = up(rec, contour_hr.to(device)).cpu()
                rec_hr_list.append(rec_hr)
                rec = rec.cpu()
                rec_list.append(rec)
            
            display1_columns = [
                image,
                einops.repeat(contour, 'B 1 H W -> B 3 H W'),
                *rec_list
            ]
            display1_image = torch.clamp(torch.stack(display1_columns, 1) + 0.5, 0, 1)
            display1_image = einops.rearrange(display1_image, 'row col ... -> (row col) ...')
            display1_image = make_grid(display1_image, nrow=len(display1_columns))
            
            display2_columns = [
                F.upsample_nearest(image, image_size),
                einops.repeat(F.upsample_nearest(contour, image_size), 'B 1 H W -> B 3 H W'),
                *rec_hr_list
            ]
            display2_image = torch.clamp(torch.stack(display2_columns, 1) + 0.5, 0, 1)
            display2_image = einops.rearrange(display2_image, 'row col ... -> (row col) ...')
            display2_image = make_grid(display2_image, nrow=len(display2_columns))
            
            return display1_image, display2_image
        
        def save_checkpoint(suffix=''):
            name = f'checkpoint{suffix}.pt'
            torch.save({
                'up': up.state_dict(),
                'opt': opt.state_dict(),
                'it': it
            }, os.path.join(up_output_dir, name))
        
        def train_step(image, contour, image_hr, contour_hr):
            image = image.to(device) # 126
            contour = contour.to(device) # 128
            image_hr = image_hr.to(device) # 256 or 512
            contour_hr = contour_hr.to(device) # 256 or 512
            
            net.eval()
            with torch.no_grad():
                rec = net(contour, image)
                
            log_dict = {}
            loss = 0

            up.train()
            mse_loss = F.mse_loss(up(rec, contour_hr), image_hr)
            log_dict['mse_loss'] = mse_loss.item()
            loss = loss + mse_loss
                
            opt.zero_grad()
            loss.backward()
            opt.step()
            net.zero_grad()
            
            return log_dict

        try:
            with tqdm(infinite_loop(up_train_loader), total=end_iteration-it) as loader:
                for image, contour, image_hr, contour_hr in loader:
                    if isinstance(end_iteration, int) and it >= end_iteration:
                        raise KeyboardInterrupt
                    
                    log_dict = train_step(image, contour, image_hr, contour_hr)
                    loader.set_postfix({'it': it, **log_dict})
    
                    if it % log_interval == 0:
                        for key, value in log_dict.items():
                            writer.add_scalar(f'train/{key}', value, it)
                    
                    if (it < 1000 and it % 200 == 0) or (it >= 1000 and it % sample_interval == 0):
                        for i, img in enumerate(display_sample()):
                            writer.add_image(f'sample/upsample{i}', img, it)
                    
                    it += 1
                    
                    if it % save_interval == 0:
                        save_checkpoint()
                    
                    if it in save_iterations:
                        save_checkpoint(suffix=f'_{it}')

        except KeyboardInterrupt:
            print('saving checkpoint')
            save_checkpoint()

    
if __name__ == '__main__':
    import fire
    fire.Fire(Main)
