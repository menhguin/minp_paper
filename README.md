# Min P: Code and Documentation
✨Hi there! ✨

This is the repository for the Min P paper! Here, you will find the following:
- **Min P Code Implementation:** The latest implementation of Min P sampling from the Huggingface Transformers library as of June 2024.
- **WandB logs of GPQA and GSM8K evals:** Logs comparing results between Min P and Top P for both GPQA and GSM8K evaluations, at different truncation sampling parameters and temperature scaling values.

## External links:
- **Colab notebook to replicate GPQA and GSM8K evals:** If you’d like to replicate the GPQA and GSM8K COT evaluations in the paper, you may do so at [PUBLIC]_Min_P_Evals_Replication_for_GPQA_and_GSM8K_COT.ipynb
- **Logs for AlpacaEval Creative Writing:** For logs of the independently run AlpacaEval Creative Writing evals for Min P, see [https://github.com/IlyaGusev/quest](https://github.com/IlyaGusev/quest) _(not affiliated with authors)_
- **Interactive Demo:** For the independently created interactive demo, check out [https://artefact2.github.io/llm-sampling/index.xhtml](https://artefact2.github.io/llm-sampling/index.xhtml) _(not affiliated with authors)_

## How to use Min P

1. **Please check if Min P is already available.** Currently it is already available on Transformers, VLLM and many others. Transformers has already merged Min P a few months back: https://github.com/huggingface/transformers/pull/30639

To use it, you only need to add a gen/output hyperparameter like you would with top_p or temperature (I think).

```
# Generate text using Top-p sampling
output = model.generate(
    input_ids,
    do_sample=True,        # Enable sampling
    top_p=0.9,             # Cumulative probability threshold
    min_p=0.1
    max_length=50          # Maximum length of generated text
)
```

2. **To integrate your own custom samplers,** you can check out the changes in the above PR to see what you need to get it working. The actual implementation which we copied into the paper is under logits_process.py, but you'd need to change a lot of other files which reference logits_process.py: https://github.com/huggingface/transformers/blob/80f2b1610fa17ebf582897c8611180cac38652f0/src/transformers/generation/logits_process.py#L4 . What you need to change entirely depends on how the inference engine is set up. For VLLM, changes were much simpler: https://github.com/vllm-project/vllm/pull/1642
3. **Do note that our evaluations were conducted on VLLM.** This is important because VLLM does temperature scaling before truncation sampling, whereas Hugging Face does the reverse order. This means you will see different behaviour depending on what you use. I recommend VLLM due to its faster speed and because diversity from temperature is higher if you do it before truncation (for creative writing for example). You will probably get better benchmark scores from Hugging Face, but I feel it sort of defeats the purpose of using temperature sampling at all.

Let me know if you have other questions!
