class Params:
    num_features: int = 80
    min_time_stretch: float = 0.9
    max_time_stretch: float = 1.1
    sample_rate: int = 16000
    min_shift: int = -3
    max_shift: int = 3
    original_sample_rate: int = 22050
    dataset: str = "LS"
    bpe_model: str = "vocabulary_LS.model" if dataset == "LS" else "vocabulary_LJ.model"
    vocab_size: int = 10000
    time_masking: int = 1
    noise_variance: float = 0.01
    batch_size: int = 8
    num_workers: int = 8
    lr: float = 0.002
    num_epochs: int = 10
    clip_grad_norm: float = 10.0
    wandb_name: str = "streaming-transformer"
    from_pretrained: bool = False
    model_path: str = "left20_right20_segment100_epoch5.pth"
    device: str = "cuda:3"
    start_epoch: int = 0
    wandb_log: bool = True
    segment_size: int = 100
    left_context: int = 64
    right_context: int = 64
    data: str = "/data/aotabisheva/data"
    config_yaml: str = "config.yaml"
    max_memory_size: int = -1
    simul_type: str = "waitk_fixed_pre_decision"
    train_subset: str = "train"
    sentence_avg: bool = False
    label_smoothing: float = 0.0
    encoder_normalize_before: bool = True
    decoder_layers: int = 3
    encoder_layers: int = 6
    encoder_ffn_embed_dim: int = 1024
    dropout: float = 0.1
    seed: int = 221


class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0
    pad_value: float = -11.5129251