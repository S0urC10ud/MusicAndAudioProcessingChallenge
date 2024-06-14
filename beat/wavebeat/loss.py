import torch


class BCFELoss(torch.nn.Module):
    def __init__(self):
        super(BCFELoss, self).__init__()

    def forward(self, input, target):
        # split out the channels
        beat_act_target = target[:,0,:]
        downbeat_act_target = target[:,1,:]

        beat_act_input = input[:,0,:]
        downbeat_act_input = input[:,1,:]

        # beat errors
        target_beats = beat_act_target[beat_act_target == 1]
        input_beats =  beat_act_input[beat_act_target == 1]

        beat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_beats, target_beats)

        # no beat errors
        target_no_beats = beat_act_target[beat_act_target == 0]
        input_no_beats = beat_act_input[beat_act_target == 0]

        no_beat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_no_beats, target_no_beats)

        # downbeat errors
        target_downbeats = downbeat_act_target[downbeat_act_target == 1]
        input_downbeats = downbeat_act_input[downbeat_act_target == 1]

        downbeat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_downbeats, target_downbeats)

        # no downbeat errors
        target_no_downbeats = downbeat_act_target[downbeat_act_target == 0]
        input_no_downbeats = downbeat_act_input[downbeat_act_target == 0]

        no_downbeat_loss = torch.nn.functional.binary_cross_entropy_with_logits(input_no_downbeats, target_no_downbeats)

        # sum up losses
        total_beat_loss = 1/2 * ((beat_loss + no_beat_loss )**2 + (beat_loss - no_beat_loss)**2)
        total_downbeat_loss = 1/2 * ((downbeat_loss + no_downbeat_loss )**2 + (downbeat_loss - no_downbeat_loss)**2)

        # find form
        total_loss = total_beat_loss + total_downbeat_loss

        return total_loss, total_beat_loss, total_downbeat_loss