# Limitations (to be stated in the manuscript)

1. **Single dataset, single split, single seed**  
   Confidence intervals reflect test-set sampling variability only, not training variability or multi-centre generalisation.

2. **Baseline models**  
   Swin-UNet, ResUNet++, DCNU-AttUNet, Standard U-Net and DeepLabV3+ were trained in separate notebooks. Their reported numbers are point estimates. Confusion matrices marked † were reconstructed from reported accuracy.

3. **Ablation protocol**  
   Ablation experiments used a reduced resolution (256 × 256) and fewer epochs to stay within compute budget. Absolute numbers are therefore lower than the 512 × 512 headline results; only *relative* changes should be interpreted.

4. **No external validation**  
   The study uses a single internal cohort. Fundus-based CIMT estimation remains a research prototype and is **not** a clinical decision tool.

5. **Annotation noise**  
   Vessel and artery–vein reference masks come from the dataset’s own annotation pipeline and may contain residual errors.

6. **Class imbalance**  
   The training set is imbalanced for the CIMT classification label (≈ 73 % positive). Threshold selection and metrics that are robust to imbalance (AUC, MCC) are therefore emphasised.
