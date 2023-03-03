# Adapted from https://github.com/archinetai/audio-diffusion-pytorch/issues/51#issuecomment-1451458653

import torch
import torchaudio
import gc
import argparse
from tqdm import tqdm
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from data_loader import DataLoader

SAMPLE_RATE = 30000
BATCH_SIZE = 1
NUM_SAMPLES = 2**18


def create_model(text_condition = False):
	if text_condition:
		return DiffusionModel(
			net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
			in_channels=1, # U-Net: number of input/output (audio) channels
			channels=[8, 32, 64, 128, 256], # U-Net: channels at each layer
			factors=[1, 4, 4, 2, 2], # U-Net: downsampling and upsampling factors at each layer
			items=[1, 2, 2, 2, 4], # U-Net: number of repeating items at each layer
			attentions=[0, 0, 0,  1, 1], # U-Net: attention enabled/disabled at each layer
			attention_heads=4, # U-Net: number of attention heads per attention item
			attention_features=32, # U-Net: number of attention features per attention item
			diffusion_t=VDiffusion, # The diffusion method used
			sampler_t=VSampler, # The diffusion sampler used
			use_text_conditioning=True, # U-Net: enables text conditioning (default T5-base)
			use_embedding_cfg=True, # U-Net: enables classifier free guidance
			embedding_max_length=64, # U-Net: text embedding maximum length (default for T5-base)
			embedding_features=768, # U-Net: text mbedding features (default for T5-base)
			cross_attentions=[0, 0, 1, 1, 1], # U-Net: cross-attention enabled/disabled at each layer
		)
	else:
		return DiffusionModel(
			net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
			in_channels=1, # U-Net: number of input/output (audio) channels
			channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
			factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
			items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
			attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
			attention_heads=8, # U-Net: number of attention heads per attention item
			attention_features=64, # U-Net: number of attention features per attention item
			diffusion_t=VDiffusion, # The diffusion method used
			sampler_t=VSampler, # The diffusion sampler used
		)


def main(args = None):
	if args is None:
		args = {
			# DataLoader args
			"csvPath": "TODO",
			"audioDirPath": "TODO",
			"audioColumn": "TODO",
			"targetColumn": "TODO",
			"samplingRate": SAMPLE_RATE,
			"numSamples": NUM_SAMPLES,
			"batchSize": BATCH_SIZE,
			"returnDataset": False,
			"shuffle": True,

			# Output args
			"resDirPath": "TODO",
			
		}
	if "resDirPath" not in args.keys():
		args["resDirPath"] = "./"

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	dataloader = DataLoader(
		csvPath=args["csvPath"],
		audioDirPath=args["audioDirPath"],
		audioColumn=args["audioColumn"],
		targetColumn=args["targetColumn"],
		samplingRate=args["samplingRate"],
		numSamples=args["numSamples"],
		batchSize=args["batchSize"],
		returnDataset=args["returnDataset"],
		shuffle=args["shuffle"]
	)

	model = create_model(text_condition=True).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

	print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

	epoch = 0
	step = 0
	
	scaler = torch.cuda.amp.GradScaler()

	checkpoint_path = args["resDirPath"] + "checkpoint-audio-diffusion.pt"

	model.train()
	while epoch < 100:
		avg_loss = 0
		avg_loss_step = 0
		progress = tqdm(range(dataloader.numBatch))
		for i in progress:
			audio, caption = dataloader.nextBatch()
			with torch.autograd.set_detect_anomaly(True):
				optimizer.zero_grad()
				audio = torch.from_numpy(audio).to(device)
				with torch.cuda.amp.autocast():
					loss = model(audio, text=caption.tolist())
					avg_loss += loss.item()
					avg_loss_step += 1
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				progress.set_postfix(
					loss=loss.item(),
					epoch=epoch + i / dataloader.numBatch,
				)

				if step % 200 == 0:
					# Turn noise into new audio sample with diffusion
					noise = torch.randn(1, 1, NUM_SAMPLES, device=device)
					with torch.cuda.amp.autocast():
						sample = model.sample(noise, num_steps=100)

					torchaudio.save(args["resDirPath"] + f'test_generated_sound_{step}.wav', sample[0].cpu(), SAMPLE_RATE)
					del sample
					gc.collect()
					torch.cuda.empty_cache()
				
				if step % 100 == 0:
					print(f"Step {step}, Epoch {epoch + i / dataloader.numBatch}, loss {avg_loss / avg_loss_step}")
					avg_loss = 0
					avg_loss_step = 0
				
				step += 1

		epoch += 1
		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
		}, checkpoint_path)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint", type=str, default=None)
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--run_id", type=str, default=None)
	return parser.parse_args()


if __name__ == "__main__":
	main()