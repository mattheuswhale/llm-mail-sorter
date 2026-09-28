
def main() -> None:
    import torch

    print("torch:", torch.__version__)
    print("torch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
