# AlexTTS

## Project Structure

### Core Files
- `data.py` - Contains dataset generation and handling 
- `tokenizer.py` - Implements `DacTokenizer` and `MisakiTokenizer` classes for text and audio tokenization 
- `transformer.py` - Contains the main `TTSTransformer` model implementation and its configuration class `TTSTransformerArgs`.
- `train.py` - Main training script with distributed training support, logging configuration, and training loop implementation.
- `generate.py` - Text to speech inference
- `optim.py` - Training optimizer code. Copied from the Lingua folder
- `lingua_transformer_modified.py` -  Meta Lingua attention implementation. Modified for audio generation

## Data preparation
torchrun --nproc_per_node <NUM_GPU> -m apps.tts.data

## Training
torchrun --nproc_per_node <NUM_GPU> -m apps.tts.train

## Generation
torchrun --nproc_per_node <NUM_GPU> -m apps.tts.generate

