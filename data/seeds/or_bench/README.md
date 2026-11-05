---
language: 
- en
pretty_name: "OR-Bench"
viewer: true
tags:
- llm
- alignment
- over-alignment
license: "cc-by-4.0"
dataset_info:
  - config_name: or-bench-80k
    features:
      - name: prompt
        dtype: string
      - name: category
        dtype: string
  - config_name: or-bench-hard-1k
    features:
      - name: prompt
        dtype: string
      - name: category
        dtype: string
  - config_name: or-bench-toxic
    features:
      - name: prompt
        dtype: string
      - name: category
        dtype: string
configs:
  - config_name: or-bench-80k
    data_files:
      - split: train
        path: or-bench-80k.csv
  - config_name: or-bench-hard-1k
    data_files:
      - split: train
        path: or-bench-hard-1k.csv
  - config_name: or-bench-toxic
    data_files:
      - split: train
        path: or-bench-toxic.csv
task_categories:
- text-generation
- question-answering
---
# OR-Bench: An Over-Refusal Benchmark for Large Language Models

Please see our **demo** at [HuggingFace Spaces](https://huggingface.co/spaces/bench-llm/or-bench). 

## Overall Plots of Model Performances
Below is the overall model performance. X axis shows the rejection rate on OR-Bench-Hard-1K and Y axis shows the rejection rate on OR-Bench-Toxic. The best aligned model should be on the top left corner of the plot where the model rejects the most number of toxic prompts and least number of safe prompts. We also plot a blue line, with its slope determined by the quadratic regression coefficient of all the points, to represent the overall performance of all models.

<img src="images/overall_x_y_plot.png" alt="Image 1" style="width: 100%;"/>

## Overall Workflow
Below is the overall workflow of our pipeline. We automate the process of producing seemingly toxic prompts that is able to produce updated prompts constantly.
<img src="images/overall_workflow.png" alt="Image 1" style="width: 100%;"/>



## Detailed Model Performance
Here are the radar plots of different model performances. The <span style="color: red;">red</span> area indicates the rejection rate of seemingly toxic prompts and the <span style="color: blue;">blue</span> area indicates the acceptance rate of toxic prompts. In both cases, the plotted area is the smaller the better.
### Claude-2.1
<div style="display: flex; flex-direction: row; justify-content: flex-start;">
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Claude-2.1.png" alt="Image 3" style="width: 100%;"/>
    <div>Claude-2.1</div>
  </div>
</div>

### Claude-3 Model Family
<div style="display: flex; flex-direction: row; justify-content: flex-start;">
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Claude-3-haiku.png" alt="Image 1" style="width: 100%;"/>
    <div>Claude-3-Haiku</div>
  </div>
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Claude-3-sonnet.png" alt="Image 2" style="width: 100%;"/>
    <div>Claude-3-Sonnet</div>
  </div>
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Claude-3-opus.png" alt="Image 3" style="width: 100%;"/>
    <div>Claude-3-Opus</div>
  </div>
</div>

### Gemini Model Family
<div style="display: flex; flex-direction: row; justify-content: flex-start;">
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Gemma-7b.png" alt="Image 2" style="width: 100%;"/>
    <div>Gemma-7b</div>
  </div>
</div>

<div style="display: flex; flex-direction: row; justify-content: flex-start;">
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Gemini-1.0-pro.png"" alt="Image 1" style="width: 100%;"/>
    <div>Gemini-1.0-pro</div>
  </div>
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Gemini-1.5-flash-latest.png"" alt="Image 1" style="width: 100%;"/>
    <div>Gemini-1.5-flash</div>
  </div>
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Gemini-1.5-pro-latest.png"" alt="Image 1" style="width: 100%;"/>
    <div>Gemini-1.5-pro</div>
  </div>
</div>

### GPT-3.5-turbo Model Family
<div style="display: flex; flex-direction: row; justify-content: flex-start;">
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/GPT-3.5-turbo-0301.png" alt="Image 1" style="width: 100%;"/>
    <div>GPT-3.5-turbo-0301</div>
  </div>
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/GPT-3.5-turbo-0613.png" alt="Image 2" style="width: 100%;"/>
    <div>GPT-3.5-turbo-0613</div>
  </div>
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/GPT-3.5-turbo-0125.png" alt="Image 3" style="width: 100%;"/>
    <div>GPT-3.5-turbo-0125</div>
  </div>
</div>

### GPT-4 Model Family
<div style="display: flex; flex-direction: row; justify-content: flex-start;">
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/GPT-4-0125-preview.png" alt="Image 1" style="width: 100%;"/>
    <div>GPT-4-0125-preview</div>
  </div>
  <!-- <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/GPT-4-1106-preview.png" alt="Image 3" style="width: 100%;"/>
    <div>GPT-4-1106-preview</div>
  </div> -->
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/GPT-4o.png" alt="Image 3" style="width: 100%;"/>
    <div>GPT-4o</div>
  </div>
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/GPT-4-turbo-2024-04-09.png" alt="Image 3" style="width: 100%;"/>
    <div>GPT-4-1106-preview</div>
  </div>
</div>

### Llama-2 Model Family
<div style="display: flex; flex-direction: row; justify-content: flex-start;">
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Llama-2-7b.png" alt="Image 1" style="width: 100%;"/>
    <div>Llama-2-7b</div>
  </div>
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Llama-2-13b.png" alt="Image 2" style="width: 100%;"/>
    <div>Llama-2-13b</div>
  </div>
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Llama-2-70b.png" alt="Image 3" style="width: 100%;"/>
    <div>Llama-2-70b</div>
  </div>
</div>

### Llama-3 Model Family
<div style="display: flex; flex-direction: row; justify-content: flex-start;">
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Llama-3-8b.png" alt="Image 1" style="width: 100%;"/>
    <div>Llama-3-8b</div>
  </div>
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Llama-3-70b.png" alt="Image 3" style="width: 100%;"/>
    <div>Llama-3-70b</div>
  </div>
</div>

### Mistral Model Family
<div style="display: flex; flex-direction: row; justify-content: flex-start;">
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Mistral-small-latest.png" alt="Image 1" style="width: 100%;"/>
    <div>Mistral-small-latest</div>
  </div>
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Mistral-medium-latest.png" alt="Image 2" style="width: 100%;"/>
    <div>Mistral-medium-latest</div>
  </div>
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Mistral-large-latest.png" alt="Image 3" style="width: 100%;"/>
    <div>Mistral-large-latest</div>
  </div>
</div>

### QWen Model Family
<div style="display: flex; flex-direction: row; justify-content: flex-start;">
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Qwen-1.5-7B.png" alt="Image 1" style="width: 100%;"/>
    <div>Qwen-1.5-7B</div>
  </div>
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Qwen-1.5-32B.png" alt="Image 2" style="width: 100%;"/>
    <div>Qwen-1.5-32B</div>
  </div>
  <div style="flex: 0 0 31%; text-align: center;">
    <img src="images/Qwen-1.5-72B.png" alt="Image 3" style="width: 100%;"/>
    <div>Qwen-1.5-72B</div>
  </div>
</div>





