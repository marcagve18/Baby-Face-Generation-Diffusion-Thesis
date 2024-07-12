# Baby Face Synthesis and Editing with Transfer Learning

This repository contains the code and resources for the thesis project on generating and editing photo-realistic baby faces using transfer learning techniques with a pre-trained diffusion model (DM). The project explores generating high-quality baby faces conditioned by facial features, ethnicity, and expression through textual prompts. It also introduces novel pipelines to modify non-identity attributes like expressions and pose orientations.

<p float="left">
  <img src="https://github.com/marcagve18/Baby-Face-Generation-Diffusion-Thesis/assets/78203527/724c7e8f-a114-46c7-a3e7-8c00eb05138e" width="30%" />
  <img src="https://github.com/marcagve18/Baby-Face-Generation-Diffusion-Thesis/assets/78203527/28a9eb50-09bd-4873-894c-018d9fb66298" width="50%" /> 
</p>
<img width="603" alt="Screenshot 2024-07-09 at 10 51 27" src="https://github.com/marcagve18/Baby-Face-Generation-Diffusion-Thesis/assets/78203527/f29511d7-2aab-4f10-b30b-e07ebdf8b17f">


### Key Findings:
- Our model achieves high realism, with 61.1% of participants unable to distinguish real baby faces from AI-generated ones.
- The approach demonstrates the effectiveness of DMs in generating and editing baby faces.

## Motivation

Generating realistic human faces is crucial for various applications such as advertising, film production, data augmentation, and medical imaging. However, generating realistic baby faces poses unique challenges due to the lack of comprehensive datasets and specific research in this area. Our project addresses these challenges by leveraging state-of-the-art generative models and transfer learning techniques.

## Objectives

1. **Generate realistic baby faces from textual descriptions.**
2. **Enable modifications to various aspects of the generated faces, such as expressions and spatial orientation.**

## Usage

For face generation and edition purposes, we have developed a package that can be found at `baby_face_generation_package`. See a basic example in `example.py`. For the orientation modification pipeline, an example can be found at `scripts/orientation-modification/ip-adapter-2.py`. 

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or inquiries, please contact [marcaguilar.1803@gmail.com](mailto:marcaguilar.1803@gmail.com).

---

We hope this project provides valuable insights and tools for researchers and practitioners working on realistic face synthesis.
