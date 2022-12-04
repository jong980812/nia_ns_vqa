import sys
sys.path.append("/data/jongseo/lab/ns-vqa/scene_parse/attr_net")
from options import get_options
from datasets import get_dataloader
from model import get_model
from trainer import get_trainer
from ddp import launch
def main():
    opt = get_options('train')
    train_loader = get_dataloader(opt, 'train')
    val_loader = get_dataloader(opt, 'val')
    model = get_model(opt)
    trainer = get_trainer(opt, model, train_loader, val_loader)

trainer.train() 
if __name__ == "__main__":
    launch(
        main,
        opt.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
