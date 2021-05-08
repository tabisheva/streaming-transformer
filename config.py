class Paramso   
num_features: int = 80
    min_time_stretch: float = 0.9
    max_time_stretch: float = 1.1
    sample_rate: int = 16000
    min_shift: int = -3
    max_shift: int = 3
    dataset: str = "LJ"
    vocab_size: int = 1000
    time_masking: int = 1
    bpe_model: str = f"vocabulary_LS_{vocab_size}.model" if dataset == "LS" else f"vocabulary_LJ_{vocab_size}.model"
    noise_variance: float = 0.01
    batch_size: int = 16
    num_workers: int = 8
    lr: float = 0.002
    num_epochs: int = 30
    clip_grad_norm: float = 10.0
    wandb_name: str = "streaming-transformer"
    from_pretrained: bool = False
    device: str = "cuda:0"
    start_epoch: int = 0
    wandb_log: bool = True
    segment_size: int = 100
    left_context: int = 32
    right_context: int = 32
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
    dropout: float = 0.25
    seed: int = 221
    linear_attention: bool = True


class MelSpectrogramConfig:
    sr: int = 16000
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0
    pad_value: float = -11.5129251
