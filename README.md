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

