Before generating any code, first evaluate whether a fully open-source implementation of each component is realistically achievable on Kaggle GPUs. If a component is not technically feasible, propose the closest practical alternative and explain the tradeoffs.


You are a principal machine learning engineer, audio AI researcher, MLOps engineer, and Python developer.

Your task is to design and build a COMPLETE end-to-end AI metalcore music generation platform using only free and open-source models and Kaggle's free GPU environment.

========================================
PROJECT GOAL
========================================

I want a system that can generate original metalcore songs inspired by:

- Currents
- Fit For A King
- Oceans Ate Alaska
- Wind Walkers
- Aviana
- The Plot In You

The goal is NOT to clone copyrighted songs.

The goal is for the model to learn:

- modern metalcore structure
- breakdowns
- ambient sections
- rhythm guitar writing
- lead guitar textures
- bass movement
- drum grooves
- emotional transitions
- tension and release

The system should ultimately generate:

1. Instrumental music
2. Lyrics
3. Vocal performances
4. Final rendered songs

========================================
SYSTEM ARCHITECTURE
========================================

Design a pipeline consisting of:

STAGE 1
Dataset Processing

Input:
WAV, MP3, FLAC

Output:
Clean training dataset

The preprocessing system must:

- validate audio files
- remove corrupt files
- normalize loudness
- convert sample rates
- split long tracks into chunks
- generate metadata
- create train/validation splits
- log errors

========================================
STAGE 2
Music Generation Model
========================================

Research available open-source models and select the BEST option.

Compare:

- MusicGen
- AudioCraft
- Stable Audio Open
- Any newer open-source alternatives

Selection criteria:

- quality
- trainability
- Kaggle compatibility
- VRAM requirements
- support for LoRA
- generation length
- community support

Choose the best model and explain why.

Implement:

- LoRA fine tuning
- mixed precision training
- checkpoint saving
- resume training
- validation generation

Target:
Train on Kaggle GPUs.

========================================
STAGE 3
Metalcore Lyrics Generator
========================================

Create a lyric generation system.

The model should learn themes such as:

- depression
- recovery
- addiction
- self destruction
- hope
- betrayal
- resilience
- emotional struggle

Use entirely open-source models.

Implement:

- dataset creation
- fine tuning pipeline
- inference pipeline

Output lyrics in:

Verse
Chorus
Verse
Bridge
Breakdown
Final Chorus

format.

========================================
STAGE 4
Vocal Generation
========================================

Integrate RVC.

Build a complete voice training pipeline.

Dataset preparation:

- isolate vocals
- organize datasets
- validate samples
- remove bad cuts

Training:

- RVC training scripts
- Kaggle compatible
- checkpoint exports

Inference:

Lyrics
→ TTS
→ RVC
→ Metalcore voice output

Support:

- screams
- aggressive vocals
- clean singing

========================================
STAGE 5
Song Assembly
========================================

Create an automated rendering pipeline.

MusicGen Instrumental
        +
Generated Lyrics
        +
TTS Vocals
        +
RVC Voice Conversion

↓

Final Song

System must automatically:

- align timing
- mix tracks
- normalize loudness
- export WAV
- export MP3

========================================
KAGGLE REQUIREMENTS
========================================

Everything must run using Kaggle free resources.

Optimize for:

- T4 GPUs
- P100 GPUs
- limited storage
- notebook session timeouts

Implement:

- automatic checkpoint saving
- dataset caching
- memory optimization
- gradient checkpointing

========================================
DELIVERABLES
========================================

Create ALL files.

Required output:

project/

├── dataset_tools/
├── music_training/
├── lyric_training/
├── rvc_training/
├── inference/
├── notebooks/
├── configs/
├── outputs/
├── README.md

Generate:

1. Complete folder structure
2. Every Python file
3. Every requirements file
4. Every config file
5. Kaggle notebook
6. Training notebook
7. Inference notebook
8. Documentation
9. Troubleshooting guide
10. Hardware requirements
11. VRAM estimates
12. Expected training durations

========================================
ENGINEERING STANDARDS
========================================

No placeholder code.

No pseudocode.

No TODO sections.

No incomplete implementations.

Generate production-ready code.

Use modern Python practices.

Type hints required.

Logging required.

Error handling required.

CLI interfaces required.

Configuration files required.

All code should run after cloning the repository and following the README.

When faced with multiple solutions:

1. Compare alternatives.
2. Select the best option.
3. Explain why.

Then generate the entire project.
