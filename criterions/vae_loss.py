from fairseq.dataclass.configs import FairseqDataclass
from torch.nn import functional as F
import torch
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("graph_rl_vae_loss", dataclass=FairseqDataclass)
class GraphRLCrossEntropy(FairseqCriterion):
    def forward(self, model, sample, reduce=True):
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        pred_trajectory, latents, feature = model(**sample["net_input"])
        pred_trajectory = pred_trajectory[:, :-1]
        data = sample["net_input"]["batched_data"]
        B, T = data['batch_size'], data['timesteps']
        target = data["seq_Y"]
        joined_inputs = data["seq_X"].to(dtype=torch.float)
        
        weights = torch.cat([
            torch.ones(2, device=joined_inputs.device) * self.task.mcfg.position_weight,
            torch.ones(self.task.mcfg.observation_dim-2, device=joined_inputs.device),
            torch.ones(self.task.mcfg.action_dim, device=joined_inputs.device) * self.task.mcfg.action_weight,
            torch.ones(1, device=joined_inputs.device) * self.task.mcfg.reward_weight,
            torch.ones(1, device=joined_inputs.device) * self.task.mcfg.value_weight,
        ])
        mse = F.mse_loss(pred_trajectory, joined_inputs, reduction='none')*weights[None, None, :]

        first_action_loss = self.task.mcfg.first_action_weight*F.mse_loss(joined_inputs[:, 0, self.task.mcfg.observation_dim:self.task.mcfg.observation_dim+self.task.mcfg.action_dim],
                                                                pred_trajectory[:, 0, self.task.mcfg.observation_dim:self.task.mcfg.observation_dim+self.task.mcfg.action_dim])
        sum_reward_loss = self.task.mcfg.sum_reward_weight*F.mse_loss(joined_inputs[:, :, -2].mean(dim=1),
                                                            pred_trajectory[:, :, -2].mean(dim=1))
        last_value_loss = self.task.mcfg.last_value_weight*F.mse_loss(joined_inputs[:, -1, -1],
                                                            pred_trajectory[:, -1, -1])
        # cross_entropy = F.binary_cross_entropy(pred_terminals, torch.clip(terminals.float(), 0.0, 1.0))
        # reconstruction_loss = (mse*mask*terminal_mask).mean()+cross_entropy
        reconstruction_loss = (mse * data["mask"]).mean()
        reconstruction_loss = reconstruction_loss + first_action_loss + sum_reward_loss + last_value_loss

        #reconstruction_loss = torch.sqrt((mse * mask).sum(dim=1)).mean()

        if self.task.mcfg.ma_update:
            loss_vq = 0
        else:
            loss_vq = F.mse_loss(latents, feature.detach())
        # Commitment objective
        loss_commit = F.mse_loss(feature, latents.detach())

        loss = (reconstruction_loss + loss_vq + loss_commit).mean()
        # loss = loss.to(dtype=torch.float32)
        
        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
            # "ncorrect": ncorrect,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / sample_size, sample_size, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True