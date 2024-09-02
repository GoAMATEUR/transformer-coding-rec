class BaseConfig:
    
    dim_embed: int = 512
    max_len: int = 1024
    dim_ffn: int = 2048
    dropout: float = 0.1    
    
    class Encoder:
        num_layers = 6
        # dim = 512 * 4
        # num_layers = 4
        # dropout = 0.1
    
    # class Decoder:
    #     dim = 512 * 4
    #     num_layers = 4
    #     dropout = 0.1
    
    class SelfAtten:
        num_heads = 8
        dim_k = 64
        dim_v = 64
        dropout = 0.1

