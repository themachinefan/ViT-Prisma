from dataclasses import dataclass
from vit_prisma.sae.config import VisionModelSAERunnerConfig
import os 
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.sae.sae import SparseAutoencoder
from custom_diffusion_pipeline import CustomSDPipeline
from PIL import Image
import torch 
from functools import partial
import einops
torch.set_grad_enabled(False)
@dataclass
class EvalConfig(VisionModelSAERunnerConfig):
    sae_path: str = "F:/ViT-Prisma_fork/data/diffusion/sae_weights/827d9bc7-openai-clip-vit-large-patch14-expansion-16-layer-20/n_images_650004.pt"
    #sae_path:str = = "F:/ViT-Prisma_fork/data/diffusion/sae_weights/827d9bc7-openai-clip-vit-large-patch14-expansion-16-layer-20/n_images_390012.pt"
    model_name: str = "openai/clip-vit-large-patch14"
    model_type: str =  "clip"
    patch_size: str = 14

    dataset_path: str = "F:/prisma_data/imagenet-object-localization-challenge"
    dataset_train_path: str = "F:/prisma_data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"
    dataset_val_path: str = "F:/prisma_data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val"
    verbose: bool = True

    device: bool = 'cuda'

    hook_point:str = "blocks.20.hook_resid_post"

FEATURES = [
    (3235, 7.118, 'Plain background with single object in foreground'),
    (14145, 5.98, 'Plain background with single object in foreground v2'),
    #(15378, 3.008, 'brown scales'),
   # (4011, 5.4, 'wine'),
   # (15598, 4.5, 'chinese characters'),
   # (5122, 5.9, 'graduation hat and maybe gown'),
   # (15128, 7.2, 'grand piano'),
  #  (16336, 10.9, 'spider in web'),
   # (4790, 6.549, 'rotary phone'),
    (15958, 1.539, 'dog leash'),
    (9797, 4.684, 'animal in snow'),
    (530, 2.963, 'animal in grass'),
    (15376, 2.566, 'pointy dog ears'),
    (13824, 4.5, 'dog with terrier style hair'),
    (5776, 3.768, 'chain link fence'),
    #(4653, 5.243, 'close up of fruit'),
    (4118, 3.701, 'water background'),
    (12449, 1.6, 'text over image'),
    (2745, 1.2, 'animal biting'),
    (15451, 0.9, 'two or more dogs maybe playing'),


]


DIFFUSION_CACHE_DIR = r"F:\ViT-Prisma_fork\data\diffusion\huggingfacecache"
INPUT_IMAGES = r"F:\ViT-Prisma_fork\data\diffusion\steering\input_images"
OUTPUT = r"F:\ViT-Prisma_fork\data\diffusion\steering\outputrandom"
DO_RANDOM=True
os.makedirs(OUTPUT,exist_ok=True)
cfg = EvalConfig()


model = HookedViT.from_pretrained(cfg.model_name, is_timm=False, is_clip=True).to(cfg.device)

sparse_autoencoder = SparseAutoencoder(cfg).load_from_pretrained(cfg.sae_path)
sparse_autoencoder.to(cfg.device)
sparse_autoencoder.eval()  

sd_pipeline = CustomSDPipeline(diffusion_cache_dir=DIFFUSION_CACHE_DIR, device= cfg.device )



#TODO multiple features in one batch, for now doing one feature at a time



# hook for the transformer
def steering_hook_full(x, hook, sparse_autoencoder, feature, multiplier, max_activation):

    reconstruction = sparse_autoencoder(x)[0]
    error = x - reconstruction

    boosted_feature_acts = sparse_autoencoder.encode_standard(x)
    #print("HMM", boosted_feature_acts[:,:,feature])
    #boosted_feature_acts[:,:,feature] = boosted_feature_acts[:,:,feature]*multiplier*max_activation
    boosted_feature_acts[:,:,feature] = multiplier*max_activation

    boosted_sae_out = einops.einsum(
            boosted_feature_acts,
            sparse_autoencoder.W_dec,
            "... d_sae, d_sae d_in -> ... d_in",
        ) + sparse_autoencoder.b_dec
    

    boosted_sae_out = sparse_autoencoder.run_time_activation_norm_fn_out(boosted_sae_out)

    return boosted_sae_out + error




images = []
for i, image_filename in enumerate(os.listdir(INPUT_IMAGES)):
    image_path = os.path.join(INPUT_IMAGES, image_filename)

    image = Image.open(image_path)
    output = os.path.join(OUTPUT, f"image_{i}")
    os.makedirs(output,exist_ok=True)
    image.resize((244,244)).save(os.path.join(output, f"00000_input_image.png"))
    image = sd_pipeline.prep_image(image)

    images.append(image)

images = torch.cat(images,dim=0)

default_embeddings = model(images)

def run_and_save(sdp:CustomSDPipeline, emb,root, name, guidance_scale=3):
    images = sdp.diffuser(emb, guidance_scale=guidance_scale)
    for i, img in enumerate(images):
        output = os.path.join(root, f"image_{i}")
        os.makedirs(output,exist_ok=True)
        img.save(os.path.join(output, f"{name}.png"))


NUM_EXAMPLES = 2


if DO_RANDOM:
    from vit_prisma.utils.data_utils.imagenet_utils import setup_imagenet_paths
    from vit_prisma.dataloaders.imagenet_dataset import ImageNetValidationDataset
    from vit_prisma.transforms.open_clip_transforms import get_clip_val_transforms
    from torch.utils.data import DataLoader
    imagenet_dataset_path = r"F:/prisma_data/imagenet-object-localization-challenge"  

    data_transforms = get_clip_val_transforms()
    imagenet_paths = setup_imagenet_paths(imagenet_dataset_path)
    dataset= ImageNetValidationDataset(imagenet_paths['val'], 
                                    imagenet_paths['label_strings'], 
                                    imagenet_paths['val_labels'], 
                                    data_transforms, return_index=True
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    random_batch = next(iter(dataloader))
    random_batch = random_batch[0].to("cuda")


for multiplier in [50,10,100,1, -1, -5, 10]:
    root = os.path.join(OUTPUT,f"multiplier_{multiplier}")
    os.makedirs(output,exist_ok=True)

    run_and_save(sd_pipeline, default_embeddings, root,"0000_default")

    for example_i in range(NUM_EXAMPLES):
        for feature, max_act, descrip in FEATURES:
            #skeptical of the max activations so just setting to 5 for simplicity?
            #max_act = 5
            if DO_RANDOM:
                images = random_batch
            embeddings = model.run_with_hooks(
                images,
                fwd_hooks=[
                    (cfg.hook_point, partial(steering_hook_full, sparse_autoencoder=sparse_autoencoder, feature=feature, multiplier=multiplier, max_activation=max_act))
                ],
                clear_contexts=True
            )
            if DO_RANDOM:
                embeddings = embeddings.mean(0)[None,...]

            output_images = sd_pipeline.diffuser(embeddings, guidance_scale=3)

        
            run_and_save(sd_pipeline, embeddings,root, f"feature_{feature}_{descrip.replace(' ', '_')}_{example_i:02d}.png")







