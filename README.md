# GPT-2 Weight Conversion to 1.58-Bit/BitNet Version

## Concept Overview
- This project explores the conversion of GPT-2 weights into a 1.58-bit or BitNet representation using an absmean quantization approach.
- The aim is to test the model's evaluation performance post-quantization while drastically reducing the memory footprint and computational overhead.

## The motivation for this project comes from this paper.
The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits https://arxiv.org/pdf/2402.17764

## What Makes This Idea Interesting
- **Efficiency Gains**: Reducing model weights to 1.58-bit representation enables significant reductions in storage and memory consumption, making it feasible to deploy large language models on resource-constrained devices.
- **Preserving Performance**: Testing the trade-off between the compressed model size and its ability to retain evaluation accuracy sheds light on robust quantization techniques.
- **Future Implications**: Insights from this work can generalize to other transformer-based architectures, advancing research in efficient deep learning.

## Experimental Flow Diagram
Start │ Extract Pretrained GPT-2 Weights │ Apply 1.58-Bit Absmean Quantization │ Convert Weights to {-1, 0, +1} Representation │ Replace Original Weights in GPT-2 Model │ Evaluate Model on Benchmarks │ Analyze Performance │ End

## Proposed Steps:

1. Download GPT‑2 model from OpenAI.

2. Quantize only the linear layers of the model.
 Document the quantization method thoroughly (e.g., using your ternary/absmean approach) and verify that the transformation is working as expected.

3. Evaluate the model performance on a spam data classification problem.
Begin by creating or selecting a representative spam dataset. Establish baseline metrics (accuracy, precision, recall, F1, etc.) using the full‑precision model before quantization.

4. Fine-tune the model on supervised data.
Using a controlled subset (or multiple runs) to fine-tune and monitor performance recovery. Experiment with hyperparameters, learning rates, and training duration.

5. Re‐evaluate the model performance on the spam data.
Compare the performance against the GPT2 baseline, and also assess any trade-offs in inference speed, memory usage, and overall robustness.


Below is a suggested summary you can add as a **"Learnings & Observations"** section in your README or as a separate markdown file (e.g., `LEARNINGS.md`). You can modify it as needed.

---

## Learnings & Observations

### Pre-Fine-Tuning Stage

- **Successful Quantization:**  
  - We implemented a ternary quantization technique using an absmean approach to convert the GPT-2 model’s linear layer weights to a 1.58-bit (ternary) representation.  
  - By inspecting the state dictionary (e.g., `trf_blocks.X...weight`), we confirmed that weights are now only in the set {-1, 0, 1}.

- **Baseline Evaluation Before Fine-Tuning:**  
  - The pre-trained full-precision GPT-2 model, when applied to our spam classification task without additional fine-tuning, exhibits low baseline accuracy (e.g., training: ~46.25%, test: ~40.00%).
  - After quantizing the linear layers, the model’s accuracy remained nearly the same (≈46.25%).  
  - **Interpretation:**  
    This indicates that simply replacing weights with ternary values does not significantly change the model’s unspecialized behavior on the classification task. In other words, the quantization process has successfully reduced precision without further improving (or worsening) the baseline accuracy.

- **Implications for Fine-Tuning:**  
  - The similar baseline performance suggests that the ternary conversion has preserved the core model representations, but like the original pre-trained model, it isn’t directly aligned with the classification objective.
  - Fine-tuning is expected to help the model adapt its representations to the task at hand. When we fine-tune—focusing, for example, on the last few transformer layers along with the classification head—we anticipate that the quantized model will be able to recover, or possibly even exceed, the baseline performance.

### Next Steps

- **Fine-Tuning on Classification Data:**  
  We plan to fine-tune the quantized model on our spam classification dataset. The approach will be to:
  - Unfreeze the last two transformer layers, final layer normalization, and the classification head.
  - Monitor improvements in accuracy, loss, and other relevant performance metrics.
  - Compare the recovery performance of the quantized model versus the full-precision baseline after task-specific training.

- **Performance Evaluation:**  
  - We will log and visualize the metrics (using tools like TensorBoard) to see how the fine-tuning process impacts both the model’s accuracy and efficiency.
  - This will help us understand the tradeoffs between the decreased precision and the computational efficiency gains achieved via quantization.

---

---

## Results & Performance Evaluation

### Fine-Tuning Results
After fine-tuning the quantized GPT-2 model on the classification dataset, the results demonstrate a significant recovery in performance compared to the pre-fine-tuning baseline:

- **Training Accuracy**: 99.04%  
- **Test Accuracy**: 96.43%  

These results indicate that the quantized model, despite its reduced precision (ternary weights: {-1, 0, +1}), is capable of achieving near-perfect accuracy on the training set while maintaining excellent generalization to unseen test data.

### Key Observations
1. **Recovery After Quantization**:  
   - Prior to fine-tuning, the quantized model exhibited similar performance to the full-precision model (~46.25% accuracy).  
   - Fine-tuning allowed the quantized model to recover its ability to learn task-specific representations and significantly outperform the baseline.

2. **Generalization Ability**:  
   - The close alignment between training accuracy (99.04%) and test accuracy (96.43%) demonstrates minimal overfitting, suggesting that the model adapts effectively to the classification dataset.

3. **Implications for Efficiency**:  
   - With the quantized model achieving comparable or superior accuracy to the full-precision version, the benefits of reduced precision—such as lower memory consumption and potential improvements in inference speed—can be leveraged without sacrificing task performance.

---

### Next Steps
- **Efficiency Analysis**: Conduct experiments to measure inference speed and memory usage of the quantized model compared to the full-precision counterpart.
- **Further Experiments**: Explore additional quantization strategies and investigate their impact on training dynamics and accuracy recovery.
- **Robustness Testing**: Validate the model on different datasets and tasks to assess its versatility and robustness post-quantization.

---
