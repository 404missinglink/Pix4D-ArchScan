# ARCHSCAN Hackathon Project

## Inspiration

The UK's infrastructure has suffered from significant underinvestment over the past two decades, leaving much of it in need of urgent repairs. One of the major challenges in addressing this issue is the surveying process for potential projects:

- Requires highly skilled individuals to visit sites on foot
- Involves manual reporting on the condition of various infrastructure elements (bridges, buildings, tunnels, etc.)
- Surveyors are time-poor and difficult to replace upon retirement

**What if we could significantly streamline this process and empower surveyors and landowners?**

## What it does

ARCHSCAN revolutionizes the surveying process by:

1. Processing aerial footage of the site
2. Providing actionable summaries for the user
3. Eliminating the need for manual footage review

**Result:** Significant time savings for users, reducing the need for extensive on-site visits and examinations.

## How we built it

ARCHSCAN is a multi-agent system comprising two powerful models working in tandem:

1. Specialized fine-tuned version of Pixtral

- Fine-tuned on satellite images and VQA (Visual Question Answering) pairs
  Provides highly targeted, descriptive summaries about scenes and features in the aerial footage
- Identifies specific infrastructure elements, their condition, and notable characteristics
- Creates detailed, frame-by-frame analysis based on set criteria
  Finetuning and inference performed on Nebius-hosted NVIDIA H100 machine

2. Mistral Large 2

- Leverages its larger parameter count for advanced reasoning and synthesis
- Processes the detailed summaries from Pixtral to generate powerful, actionable insights
- Provides higher-level analysis, recommendations, and prioritization of issues
- Contextualizes the visual data within broader infrastructure management strategies

This two-stage approach allows us to combine the strengths of both models:

- Pixtral excels at extracting relevant visual information from aerial imagery
- Mistral Large 2 excels at interpreting this information and providing strategic, actionable advice

**User Interface:** The application is accessible through a Gradio UI.

### Finetuning

#### Why fine-tune for our use case?

Our focus on drone/aerial survey imagery (mostly top view) presented specific challenges:

1. Dense features in aerial imagery
2. Complex spatial context in aerial imagery
3. Difficulty understanding quantitative context as the number of features increases

> 💡 **Interesting question:** Does a VQA dataset work better for fine-tuning vision language models?

#### LoRA Finetune and Dataset

**Step 1: Baselining**

- Started with instruction-tuned offspring of Pixtral on Hugging Face: [Ertugrul/Pixtral-12B-Captioner-Relaxed](https://huggingface.co/Ertugrul/Pixtral-12B-Captioner-Relaxed)
- Improvement over "Vanilla Pixtral": [mistralai/Pixtral-12B-2409](https://huggingface.co/mistralai/Pixtral-12B-2409)

**Step 2: Fine-Tune Dataset creation**
Two-step finetuning process:

1. Flood dataset in Image Caption setup
   - We used the FloodNet Track 2 dataset: [FloodNet Track 2 on Dataset Ninja](https://datasetninja.com/floodnet-track-2)
   - This dataset provides high-quality aerial imagery of flood events, which aligns well with our focus on infrastructure surveying
2. Aerial data in Image Caption setup
   - Captions created using Pixtral with quantitative > descriptive questions
   - Created a small batch of 10 image-caption pairs
   - We also utilized a subset of images from the FGVC Aircraft dataset: [FGVC Aircraft on Hugging Face](https://huggingface.co/datasets/Multimodal-Fatima/FGVC_Aircraft_train)
   - This dataset provided additional aerial perspectives, enhancing our model's ability to interpret various types of infrastructure and objects from above

**Hardware:** Both finetuning and inference for the custom Pixtral model were performed on a Nebius-hosted NVIDIA H100 machine, providing the necessary computational power for this advanced AI task.

### Workflow

1. Custom logic to extract scenes from video as a set of frames
2. User-selectable number of frames in the demo
3. Pixtral processes individual frames
4. Complete history of summaries passed to Mistral Large 2
5. Mistral Large 2 generates final summarization

## Challenges we ran into

- Rate limiting on Mistral API
- High inference time per frame on Pixtral (10s), limiting usability with 30+ frames

## Accomplishments that we're proud of

1. **Custom Pixtral finetune:** Significantly improved summaries and accuracy, despite using a dataset not directly related to the field
2. **Practical application:** Developed a tool that can be used today on existing projects to:
   - Help surveyors and owners spot previously missed issues
   - Reduce workload and time investment, even in its hackathon state

## What we learned

- Multi-agent frameworks combining small VLM + larger LLM can be exceptionally powerful, even at zero-shot
- Finetuning Pixtral further improved results, but original outputs were already impressive
- A Mistral API-only version is available for testing: [Pix4D-ArchScan GitHub Repository](https://github.com/404missinglink/Pix4D-ArchScan/tree/UI)

## What's next for ARCHSCAN

We're excited to continue building upon ARCHSCAN:

1. **Human-in-the-loop:** Incorporate real surveyors to align outputs with industry expertise
2. **Expanded dataset:** Further finetune for even better accuracy
3. **Advanced techniques:**
   - Utilize multimodal embeddings
   - Implement RAG (Retrieval-Augmented Generation) to improve speed and quality of Mistral Large 2 summaries
4. **End goal:** Further reduce time investment and provide workers with the data they need to make informed decisions

**Note for H100:**
If you have access to an NVIDIA H100 machine, you can run inference and use our finetuned model for significantly better results. Check out our repository:
[https://github.com/404missinglink/Pix4D-ArchScan/tree/UI_Finetune](https://github.com/404missinglink/Pix4D-ArchScan/tree/UI_Finetune)

This version includes our latest improvements and optimizations for high-performance hardware, allowing you to leverage the full potential of ARCHSCAN.
