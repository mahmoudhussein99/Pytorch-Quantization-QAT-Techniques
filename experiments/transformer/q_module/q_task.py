from .utils import quant_args

from fairseq.tasks import register_task
from fairseq import utils as fairseq_utils
from fairseq.data import data_utils
from fairseq.tasks.translation import TranslationTask, TranslationConfig
from experiments.transformer.utils import tformer_replace_mha, tformer_replace_linear, tformer_replace_layer_norm
import logging
import os
import torch
import wandb

logger = logging.getLogger(__name__)


@register_task("qtranslation", dataclass=TranslationConfig)
class QTranslationTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: TranslationConfig

    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        self.cfg = cfg
        super().__init__(cfg, src_dict, tgt_dict)

    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)

        parser = quant_args(parser)

    @classmethod
    def setup_task(cls, cfg: TranslationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = fairseq_utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(
                paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly")

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang)))
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang,
                                                       len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang,
                                                       len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def build_model(self, cfg):

        model = super().build_model(cfg)

        cfg.last_error_bits = cfg.last_error_sig + cfg.last_error_man + 1
        cfg.mha_error_bits = cfg.mha_error_sig + cfg.mha_error_man + 1
        cfg.fc_error_bits = cfg.fc_error_sig + cfg.fc_error_man + 1
        cfg.mha_linear_error_bits = cfg.mha_linear_error_sig + cfg.mha_linear_error_man + 1


        tformer_replace_mha(cfg, model)
        tformer_replace_linear(cfg, model)
        tformer_replace_layer_norm(cfg, model)
        print(model)
        return model

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0

        with torch.autograd.profiler.record_function("backward"):

            if torch.isnan(loss):
                print('Training NAN LOSSSS')
                wandb.log({'NaN': 1})
                exit(-1)

            # L2 regularization for PACT
            l2_alpha = 0.0
            for name, param in model.named_parameters():
                if "alpha" in name:
                    l2_alpha += torch.norm(param)
            loss += (0.0001 * l2_alpha)

            if torch.isnan(loss):
                print('NAN LOSSSS')
                wandb.log({'NaN': 2})
                exit(-1)
            optimizer.backward(loss)

            # Adaptive gradScale
            for name, param in model.named_parameters():
                if 'adaptive_scale' in name:
                    if param.grad.data > 0:
                        param.data *= 2.0
                    elif param.grad.data < 0:
                        param.data *= 0.5
                    param.grad = None

        return loss, sample_size, logging_output
