from fairseq.optim import register_optimizer
from fairseq.optim.adam import FairseqAdam, FairseqAdamConfig


@register_optimizer("bnb", dataclass=FairseqAdamConfig)
class FairseqBnb(FairseqAdam):
    """Adam optimizer for fairseq.
    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, cfg: FairseqAdamConfig, params):
        super().__init__(cfg, params)
        if torch.cuda.is_available():
            import bitsandbytes as bnb
            self._optimizer = bnb.optim.Adam8bit(params, **self.optimizer_config)
        else:
            raise Exception('bnb only works when cuda is available')
