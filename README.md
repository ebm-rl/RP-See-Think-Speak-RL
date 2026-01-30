# Code for [Reward-Decomposed Reinforcement Learning for Immersive Video Role-Playing]

## Stage-1: CoT SFT cold-start phase

To mitigate format collapse and initialize the model with "See-Think-Speak" reasoning capabilities, execute the cold-start training script:

```bash
# Launch script for the CoT SFT cold-start phase
bash ./src/scripts/RP_COT_SFT.sh
```

## Stage-2: "See-Think-Speak" RL(GRPO) phase
To initiate the Reinforcement Learning (RL) training, execute the provided script below. Please note the following configurations for the CLIP-based Scene–Text Alignment Reward module:

* **CLIP-MAX Method**: Replace the reward calculation script with `src/open_r1/clip_MAX_server.py`.

* **CLIP-SentTopK Method**: Replace the reward calculation script with `src/open_r1/clip_Topk_server.py`.

```bash
# Launch script for the EBM-RL phase
bash ./src/scripts/EBM_RP_RL.sh
```

The following curves demonstrate the performance of the three core reward components during the EBM-RL phase: CLIP-based Alignment,PCG-based Reasoning Consistency and BERTScore-based Accuracy.

<table align="center">
  <tr>
    <td align="center"><img src="./images/clip_max.png" width="100%" alt="CLIP Reward Curve" /><br /><b>CLIP Reward</b></td>
    <td align="center"><img src="./images/pcg.png" width="100%" alt="PCG Reward Curve" /><br /><b>PCG Reward</b></td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <img src="./images/bertscore.png" width="45%" alt="BERTScore Reward Curve" /><br /><b>BERTScore Reward</b>
    </td>
  </tr>
</table>