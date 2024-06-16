# Min P: Code and Documentation
✨Hi there! ✨

This is the repository for the Min P paper! Here, you will find the following:
- **Min P Code Implementation:** The latest implementation of Min P sampling from the Huggingface Transformers library as of June 2024.
- **WandB logs of GPQA and GSM8K evals:** Logs comparing results between Min P and Top P for both GPQA and GSM8K evaluations, at different truncation sampling parameters and temperature scaling values.

# PLEASE NOTE THE BELOW LINKS MAY NOT BE ANONYMISED FOR PEER REVIEW

Min P has been very popular in the open source LLM community, and many implementations include references to the original authors.
We include links to:
- **Other implementations:** Beyond [Huggingface Transformers](https://github.com/huggingface/transformers/pull/30639), Min P has been merged into [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/3841), [VLLM](https://github.com/vllm-project/vllm/pull/1642), Kobold and many other LLM hosting/inference services.
- **Colab notebook to replicate GPQA and GSM8K evals:** If you’d like to replicate the GPQA and GSM8K COT evaluations in the paper, you may do so at  [this Google Colab Notebook.](https://colab.research.google.com/drive/1lpBoRzw273VXOECaz8AXGJlqI3wuYrEM)
- **Logs for AlpacaEval Creative Writing:** For logs of the indepedently run AlpacaEval Creative Writing evals for Min P, see https://github.com/IlyaGusev/quest
- **Interactive Demo:** For the interactive demo, check out https://artefact2.github.io/llm-sampling/index.xhtml
