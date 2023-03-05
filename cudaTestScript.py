import torch

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("device", device)
	a = torch.Tensor(5,3)
	a = a.cuda()

if __name__ == "__main__":
	main()