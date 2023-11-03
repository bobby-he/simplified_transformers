import torch
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names
import wandb
from .model_utils import myGPT2Attention, myGPT2MLP, MyConv1D, RMSNorm


class MyTrainer(Trainer):
    def create_optimizer(self):
        """
        Identical to standard HF AdamW optimizer, but with no WD for gain parameters.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(
                opt_model, [torch.nn.LayerNorm, RMSNorm]
            )
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            gain_parameters = [name for name in decay_parameters if "gain" in name]

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n in decay_parameters
                            and n not in gain_parameters
                            and p.requires_grad
                        )
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in gain_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Identical to HF transformers compute_loss, but with extra logging.
        """

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.state.global_step % 100 == 0 and "wandb" in self.args.report_to:
            if self.args.report_gains:
                to_report = {}
                for i, block in enumerate(model.transformer.h):
                    if type(block.mlp) is myGPT2MLP:
                        to_report[
                            f"{i}.mlp_block_resid_gain"
                        ] = block.mlp_block_resid_gain.data.norm()
                    if type(block.attn.v_attn) is MyConv1D:
                        to_report[
                            f"attn.{i}.value_skip_gain"
                        ] = block.attn.v_attn.skip_gain.data
                        to_report[
                            f"attn.{i}.value_resid_gain"
                        ] = block.attn.v_attn.resid_gain.data
                    if type(block.attn.c_proj) is MyConv1D:
                        to_report[
                            f"attn.{i}.proj_skip_gain"
                        ] = block.attn.c_proj.skip_gain.data
                        to_report[
                            f"attn.{i}.proj_resid_gain"
                        ] = block.attn.c_proj.resid_gain.data
                    if type(block.attn) is myGPT2Attention:
                        to_report[
                            f"attn.{i}.attn_mat_skip_gain_mean"
                        ] = block.attn.attn_mat_skip_gain.data.mean()
                        to_report[
                            f"attn.{i}.attn_mat_resid_gain_mean"
                        ] = block.attn.attn_mat_resid_gain.data.mean()
                        to_report[
                            f"attn.{i}.centre_attn_gain_mean"
                        ] = block.attn.centre_attn_gain.data.mean()
                        to_report[
                            f"attn.{i}.attn_mat_skip_gain_std"
                        ] = block.attn.attn_mat_skip_gain.data.std()
                        to_report[
                            f"attn.{i}.attn_mat_resid_gain_std"
                        ] = block.attn.attn_mat_resid_gain.data.std()
                        to_report[
                            f"attn.{i}.centre_attn_gain_std"
                        ] = block.attn.centre_attn_gain.data.std()
                wandb.log(to_report)

            if self.args.report_attn_entropy:
                to_report = {}
                for i, attn_mat in enumerate(outputs["attentions"]):
                    ent = -torch.nansum(attn_mat * torch.log(attn_mat), dim=-1)
                    to_report[f"attn.{i}.entropy"] = ent.mean()
                wandb.log(to_report)

        return (loss, outputs) if return_outputs else loss
