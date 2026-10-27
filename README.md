# AI Music Generation with MusicGen and LoRA

An experimental music-generation application built with Streamlit, Meta's MusicGen models, and parameter-efficient fine-tuning adapters from Hugging Face.

## Overview

The application provides an interface for generating instrumental music using pretrained MusicGen models and LoRA adapters.

The project explores two music-generation directions:

- **Carnatic-inspired music:** Experiments with flute and veena timbres using a fine-tuned adapter.
- **Punk/Rock music:** Experiments with a MusicGen Melody model and a fine-tuned adapter.

## Technologies

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA adapters)
- Hugging Face Hub

## Models

- Base model: `facebook/musicgen-small`
- Carnatic adapter: `fahrahmn/saraga-dreambooth-musicgen`
- Punk/Rock adapter: `fahrahmn/musicgen-melody-lora_joel`

The adapters are separate from this repository and are loaded from the Hugging Face Hub.

## Current Status

This is an experimental project. The interface runs, but audio quality and generation consistency have not been fully validated. Generated results may contain noise or sound musically incoherent.

The next development goal is to investigate model loading, adapter compatibility, audio generation, and output quality.

## Running Locally

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies from `requirements.txt`.
4. Run the application:

   ```bash
   python -m streamlit run app.py
   ```

Exact dependency versions and hardware requirements may need adjustment for your environment.

## Learning Goals

This project is part of an ongoing exploration of generative AI, transformer-based audio models, and parameter-efficient fine-tuning.

## Attribution and Licensing

The base models, datasets, and adapters have their own licenses and usage conditions. Review those terms before redistributing model files or using generated outputs commercially. The Carnatic adapter is listed under CC-BY-NC-4.0; verify the relevant model cards for current terms.

This repository is an application experiment and does not claim original ownership of the underlying models or adapters.