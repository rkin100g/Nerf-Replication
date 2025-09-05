from src.config import cfg, args
import numpy as np
import os
import torch

class TestNet(torch.nn.Module):
    def forward(self, x):
        return x

### SCRIPTS BEGINING ###
def run_dataset():
    import tqdm
    from src.datasets import make_data_loader

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    debug_flag = 0
    for batch in tqdm.tqdm(data_loader):
        if debug_flag == 0:
            print("rays_o shape:", batch["rays_o"].shape)
            print("rays_d shape:", batch["rays_d"].shape)
            print("example rays_o[0]:", batch["rays_o"][0])
            print("example rays_d[0]:", batch["rays_d"][0])
            debug_flag = 1
        pass

def run_input():
    """
    test input of network from dataset to render's starfitied sampleing 
    """
    import tqdm
    from src.datasets import make_data_loader
    from src.models.nerf.renderer import make_renderer
    from src.models.nerf.renderer.volume_renderer import Renderer

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    network = TestNet()
    renderer = make_renderer(cfg, network)
    
    debug_flag = 0
    for batch in tqdm.tqdm(data_loader):
        if debug_flag == 0:
            print("dataset output's rays_o shape:", batch["rays_o"].shape)
            print("dataset output's rays_d shape:", batch["rays_d"].shape)
            print("example rays_o[0]:", batch["rays_o"][0])
            print("example rays_d[0]:", batch["rays_d"][0])

            renderer.render(batch)
            debug_flag = 1
        pass

def run_network():
    import tqdm
    import torch
    import time
    from src.models import make_network
    from src.datasets import make_data_loader
    from src.models.nerf.renderer import make_renderer
    from src.utils.net_utils import load_network
    from src.utils.data_utils import to_cuda

    """
    print("DEBUG: CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("DEBUG: is_available =", torch.cuda.is_available())
    print("DEBUG: device_count =", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("DEBUG: get_device_name =", torch.cuda.get_device_name(0))
    """
    
    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            output = renderer.render(batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))


def run_evaluate():
    import time
    import tqdm
    import torch
    from src.datasets import make_data_loader
    from src.evaluators import make_evaluator
    from src.models import make_network
    from src.models.nerf.renderer import make_renderer
    from src.utils.net_utils import load_network

    print(f"trained_model_dir: {cfg.trained_model_dir}")
    network = make_network(cfg).cuda()
    load_network(
        network, cfg.trained_model_dir, resume=cfg.resume, epoch=cfg.test.epoch
    )
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    renderer = make_renderer(cfg, network)
    net_time = []
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != "meta":
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            output = renderer.render(batch)
            torch.cuda.synchronize()
            end_time = time.time()
        net_time.append(end_time - start_time)
        evaluator.evaluate(output, batch)
    evaluator.summarize()
    if len(net_time) > 1:
        print("net_time: ", np.mean(net_time[1:]))
        print("fps: ", 1.0 / np.mean(net_time[1:]))
    else:
        print("net_time: ", np.mean(net_time))
        print("fps: ", 1.0 / np.mean(net_time))


if __name__ == "__main__":
    globals()["run_" + args.type]()
