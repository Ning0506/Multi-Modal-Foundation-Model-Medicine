import torch
import models_mae as mm

from util.pos_embed import interpolate_pos_embed

if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"GPU(s) available: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
else:
    print("No GPU available, using the CPU instead.")

model = mm.mae_vit_small_patch16()

checkpoint = torch.load("vit-s_CXR_0.3M_mae_CheXpert.pth", map_location='cpu') 

print("Loaded pre-trained checkpoint.")

checkpoint_model = checkpoint['model']

state_dict = model.state_dict()

for k in checkpoint_model.keys():
    if k in state_dict:
        if checkpoint_model[k].shape == state_dict[k].shape:
            state_dict[k] = checkpoint_model[k]
            print(f"Loaded Index: {k} from Saved Weights")
        else:
            print(f"Shape of {k} doesn't match with {state_dict[k]}")
    else:
        print(f"{k} not found in Init Model")

# You might also need to interpolate the position embeddings if your model's configuration doesn't match the checkpoint's
interpolate_pos_embed(model, state_dict)

# Now, try loading the state_dict again
msg = model.load_state_dict(state_dict)
print(msg)
