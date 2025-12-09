# AI-Based Neural Style Transfer

## Project Information
- **Institution:** Siddaganga Institute of Technology, Tumakuru-572103
- **Department:** Computer Science & Engineering
- **Academic Year:** 2024-25
- **Batch ID:** 12

### Team Members
- Aman Kumar (1SI22CS014)u
- Ashay Amal (1SI22CS027)
- Avinash Sarraf (1SI22CS030)
- Chandragupta Kumar (1SI22CS046)

### Guide
- Guide's Name, Designation, Department of CSE, SIT Tumakuru-03

---

## Abstract

The project focuses on Neural Style Transfer (NST) to create a comprehensive system for artistic image generation. The system uses CycleGAN for style transfer, enabling the transformation of ordinary images into stunning artworks by applying the visual characteristics of renowned artists or distinctive animation styles. We implement four distinct style transfer models: One Piece, Disney, Studio Ghibli, and Van Gogh. For the animation styles (One Piece, Disney, and Ghibli), the system transforms human face photographs into their respective animated character styles. For Van Gogh style, landscape photographs are transformed into post-impressionist paintings. The datasets were curated through custom frame extraction algorithms applied to animation episodes and movies, combined with human face and landscape datasets from Kaggle. The project demonstrates the effectiveness of CycleGAN for unpaired image-to-image translation and has applications in creative design, entertainment, social media filters, education, and digital media.

---

## Chapter 1: Introduction

### 1.1 Motivation
The evolution of artificial intelligence and deep learning has opened new frontiers in creative and artistic domains. Neural Style Transfer (NST) has emerged as a technique capable of transforming ordinary images into stunning artworks by applying the visual characteristics of renowned artists or distinctive art styles.

**Key motivations:**
- Bridge the gap between artistic authenticity and computational creativity
- Create a system that stylizes images with artist-specific fidelity
- Real-world applications: personalized digital art filters, art history education tools, AI-assisted creative design

### 1.2 Objectives
1. **Artist & Style-Specific Neural Style Transfer:** Develop dedicated neural style transfer models for four distinct styles:
   - **One Piece Style:** Transform human faces into One Piece anime character style
   - **Disney Style:** Transform human faces into Disney animated character style
   - **Studio Ghibli Style:** Transform human faces into Studio Ghibli character style
   - **Van Gogh Style:** Transform landscape photographs into Van Gogh painting style

2. **Performance Optimization for Real-Time Stylization:** Optimize computational efficiency to reduce processing time to under 3 seconds per image

3. **User-Centric System Design:** Build a user-friendly interface with intuitive style selection, multiple image format support, and customization options

4. **Style Quality Evaluation:** Implement qualitative and quantitative metrics (content loss, style loss, user satisfaction score)

5. **Custom Dataset Creation:** Build comprehensive datasets through frame extraction from animation sources and curated Kaggle datasets

---

## Chapter 2: Literature Survey

| Paper | Authors | Key Contribution |
|-------|---------|------------------|
| [1] | Gatys et al. (2016) | Pioneered NST using CNNs to separate and recombine content and style |
| [2] | Johnson et al. (2016) | Introduced perceptual loss functions and feedforward network for real-time style transfer |
| [3] | Huang & Belongie (2017) | Adaptive Instance Normalization (AdaIN) for arbitrary style transfer |
| [4] | Park et al. (2019) | Attention mechanisms for improved stylization quality |
| [5] | Kotovenko et al. (2019) | Content and style disentanglement for artist-specific models |
| [6] | Li et al. (2017) | Universal style transfer using Whitening and Coloring Transforms (WCT) |
| [7] | Zhang et al. (2018) | Deep feature statistics with patch-based loss |
| [8] | Ruder et al. (2016) | Temporal coherence for video stylization |
| [9] | Jing et al. (2020) | Comprehensive survey of NST methods |
| [10] | Zhang & Dana (2020) | Style consistency network with semantic segmentation |

### Research Gaps Addressed
- Improving style fidelity for highly detailed or complex artistic styles
- Computational efficiency for real-time applications
- Artist-specific model training for authentic style reproduction

---

## Chapter 3: System Overview

### 3.1 System Overview

The system is designed to transform user-provided content images into artistic stylizations using Neural Style Transfer powered by CycleGAN.

#### Style Transfer Mode
- User uploads a content image and selects an art style/artist
- Available styles: One Piece, Disney, Studio Ghibli, Van Gogh
- Uses CycleGAN-based architecture for unpaired image-to-image translation
- Processing optimized for under 3 seconds per image on GPU
- Animation styles (One Piece, Disney, Ghibli) transform human faces into animated characters
- Van Gogh style transforms landscape photographs into post-impressionist paintings

### 3.2 Style Transfer Module
- **Architecture:** Encoder-decoder with pre-trained VGG-19 encoder
- **Approach:** Style-specific CycleGAN model trained separately for each of the four styles
- **Features captured:** Brush strokes, textures, color palettes unique to each style
- **Output:** High-resolution stylized image preserving semantic structure

#### 3.2.1 One Piece Style Characteristics
The One Piece style model captures the distinctive features of Eiichiro Oda's artwork:
- **Bold Outlines:** Strong black outlines defining character features
- **Exaggerated Proportions:** Distinctive large eyes and expressive faces
- **Vibrant Colors:** Bright, saturated color palette
- **Dynamic Expressions:** Exaggerated emotional expressions characteristic of the series
- **Shading Style:** Cell-shaded appearance with clear light and shadow areas

#### 3.2.2 Disney Style Characteristics
The Disney style model replicates the classic Disney animation aesthetic:
- **Smooth Gradients:** Soft color transitions and gradients
- **Large Expressive Eyes:** Characteristic Disney-style eyes with detailed reflections
- **Soft Color Palettes:** Warm, inviting color schemes
- **Polished Appearance:** Clean lines and refined character features
- **Realistic Proportions:** More realistic body proportions compared to anime styles

#### 3.2.3 Studio Ghibli Style Characteristics
The Ghibli style model captures the unique aesthetic of Hayao Miyazaki's films:
- **Watercolor Textures:** Soft, painterly appearance reminiscent of watercolors
- **Naturalistic Expressions:** Subtle, realistic facial expressions
- **Warm Earthy Tones:** Natural color palettes with greens, browns, and soft blues
- **Hand-Drawn Feel:** Organic lines that preserve the hand-animated aesthetic
- **Detailed Features:** Attention to fine details in hair, eyes, and facial features

#### 3.2.4 Van Gogh Style Characteristics
The Van Gogh style model reproduces the post-impressionist master's painting technique:
- **Swirling Brush Strokes:** Distinctive circular and flowing brush patterns
- **Vibrant Colors:** Bold use of yellows, blues, and greens
- **Impasto Technique:** Thick, textured appearance of paint application
- **Emotional Expression:** Colors and strokes that convey mood and emotion
- **Movement and Energy:** Dynamic brush strokes that create visual movement

### 3.3 CycleGAN Loss Functions

**Total Loss Function:**
```
L_total = L_GAN(G, D_Y, X, Y) + L_GAN(F, D_X, Y, X) + λ · L_cyc(G, F)
```

Where:
- L_GAN(G, D_Y, X, Y): Adversarial loss for mapping G: X → Y
- L_GAN(F, D_X, Y, X): Adversarial loss for mapping F: Y → X
- L_cyc(G, F): Cycle-consistency loss
- λ: Weighting factor for cycle loss

**Cycle-Consistency Loss:**
```
L_cyc(G, F) = E[||F(G(x)) - x||_1] + E[||G(F(y)) - y||_1]
```

### 3.4 Dataset Creation

Since CycleGAN requires unpaired datasets from two domains (Domain X: real-world images, Domain Y: stylized images), we curated custom datasets for each of the four style transfer models: One Piece, Disney, Studio Ghibli, and Van Gogh.

#### 3.4.1 One Piece Style Dataset (Human to One Piece Animation)

**Domain X - Human Faces:**
- **Source:** Human Faces Dataset from Kaggle
- **Content:** Real human face photographs with diverse demographics
- **Preprocessing:** Cropped and resized to 256×256 pixels

**Domain Y - One Piece Character Frames:**
- **Source:** Extracted from One Piece anime episodes and movies
- **Extraction Method:** Custom frame extraction algorithm
- **Process:**
  1. One Piece episodes and movies were processed frame by frame using OpenCV
  2. Character detection algorithm identified frames containing character faces
  3. Frames with clear character appearances were extracted
  4. Extracted frames were cropped to focus on character faces
  5. Images resized to 256×256 pixels for training
- **Characteristics:** Bold outlines, exaggerated expressions, vibrant colors, distinctive eye styles

#### 3.4.2 Disney Style Dataset (Human to Disney Animation)

**Domain X - Human Faces:**
- **Source:** Human Faces Dataset from Kaggle
- **Content:** Real human face photographs
- **Preprocessing:** Cropped and resized to 256×256 pixels

**Domain Y - Disney Character Frames:**
- **Source:** Extracted from Disney animated movies
- **Extraction Method:** Custom frame extraction algorithm
- **Process:**
  1. Disney animated movies were scanned frame by frame
  2. Character detection identified frames with character faces
  3. High-quality frames with clear character appearances were extracted
  4. Frames cropped to focus on character facial features
  5. Images resized to 256×256 pixels for training
- **Characteristics:** Smooth gradients, large expressive eyes, soft color palettes, polished animation style

#### 3.4.3 Studio Ghibli Style Dataset (Human to Ghibli Animation)

**Domain X - Human Faces:**
- **Source:** Human Faces Dataset from Kaggle
- **Content:** Real human face photographs
- **Preprocessing:** Cropped and resized to 256×256 pixels

**Domain Y - Ghibli Character Frames:**
- **Source:** Extracted from Studio Ghibli movies (Spirited Away, Howl's Moving Castle, My Neighbor Totoro, etc.)
- **Extraction Method:** Custom frame extraction algorithm
- **Process:**
  1. Studio Ghibli movies were processed frame by frame
  2. Character detection algorithm identified frames containing character faces
  3. Frames with clear character appearances were extracted
  4. Extracted frames were cropped to focus on character faces
  5. Images resized to 256×256 pixels for training
- **Characteristics:** Soft watercolor-like textures, detailed backgrounds, naturalistic expressions, warm earthy tones

#### 3.4.4 Van Gogh Style Dataset (Photo to Van Gogh Painting)

**Domain X - Real-World Landscapes:**
- **Source:** Landscape photographs from Kaggle
- **Content:** Natural scenery, landscapes, outdoor scenes
- **Preprocessing:** Resized to 256×256 pixels

**Domain Y - Van Gogh Paintings:**
- **Source:** Van Gogh Paintings Dataset from Kaggle
- **Content:** Original paintings by Vincent van Gogh (both landscapes and portraits)
- **Characteristics:** Distinctive swirling brush strokes, vibrant colors, post-impressionist style, thick impasto technique
- **Preprocessing:** Resized to 256×256 pixels

#### 3.4.5 Frame Extraction Algorithm

The following algorithm was used to extract character frames from One Piece episodes, Disney movies, and Studio Ghibli films:

```
Algorithm: Character Frame Extraction from Video

Input: Video file V, Character detection model M
Output: Set of extracted frames F

1. Initialize empty frame set F
2. Load video V using OpenCV and get total frame count
3. For each frame f in video:
   a. Apply character/face detection model M on frame f
   b. If character detected with confidence > threshold:
      i. Crop frame to bounding box of detected character
      ii. Resize cropped image to 256×256
      iii. Add to frame set F
   c. Skip duplicate/similar frames using similarity threshold
4. Return frame set F
```

**Frame Extraction Sources:**
| Animation Style | Source Material |
|-----------------|-----------------|
| One Piece | Anime episodes and movies |
| Disney | Animated feature films |
| Studio Ghibli | Animated feature films |

#### 3.4.6 Dataset Summary

| Style Model | Domain X (Content) | Domain X Source | Domain Y (Style) | Domain Y Source |
|-------------|-------------------|-----------------|------------------|-----------------|
| One Piece | Human Faces | Kaggle Dataset | One Piece Characters | Episode/Movie Frame Extraction |
| Disney | Human Faces | Kaggle Dataset | Disney Characters | Movie Frame Extraction |
| Studio Ghibli | Human Faces | Kaggle Dataset | Ghibli Characters | Movie Frame Extraction |
| Van Gogh | Landscape Photos | Kaggle Dataset | Van Gogh Paintings | Kaggle Dataset |

### 3.5 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 256 × 256 |
| Batch Size | 4 |
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Beta1 (Adam) | 0.5 |
| Beta2 (Adam) | 0.999 |
| Epochs | 200 |
| Lambda (Cycle Loss Weight) | 10 |
| Loss Functions | GAN Loss, Cycle Loss, Identity Loss |

### 3.6 Training Configuration per Style

| Style Model | Training Epochs | Domain X Size | Domain Y Size | Training Time (approx.) |
|-------------|-----------------|---------------|---------------|------------------------|
| One Piece | 200 | 3000+ | 2500+ | 12-15 hours |
| Disney | 200 | 3000+ | 2000+ | 12-15 hours |
| Studio Ghibli | 200 | 3000+ | 2000+ | 12-15 hours |
| Van Gogh | 200 | 2500+ | 400+ | 8-10 hours |

### 3.7 Training Process

#### 3.7.1 Training Pipeline
1. **Data Loading:** Images loaded from Domain X and Domain Y directories
2. **Preprocessing:** Images resized to 256×256 and normalized to [-1, 1]
3. **Batch Formation:** Random sampling of unpaired images from both domains
4. **Forward Pass:**
   - Generator G transforms X → Y (e.g., human face → animation style)
   - Generator F transforms Y → X (animation style → human face)
5. **Loss Computation:**
   - Adversarial loss computed using discriminators D_X and D_Y
   - Cycle-consistency loss ensures X → Y → X ≈ X
   - Identity loss for color preservation (optional)
6. **Backward Pass:** Gradients computed and parameters updated
7. **Checkpoint Saving:** Model weights saved every 10 epochs

#### 3.7.2 Hardware Requirements
- **Training:** NVIDIA GPU with 8GB+ VRAM (GTX 1080 Ti, RTX 2080, or better)
- **Inference:** GPU with 4GB+ VRAM for real-time processing
- **RAM:** 16GB+ system memory recommended
- **Storage:** 50GB+ for datasets and model checkpoints

#### 3.7.3 Training Environment
- **Platform:** Google Colab Pro / Local GPU workstation
- **Framework:** PyTorch 2.0+
- **CUDA Version:** 11.7+
- **Python Version:** 3.8+

---

## Chapter 4: System Architecture and High Level Design

### 4.1 Terminology
- **CycleGAN:** Generative adversarial network for unpaired image-to-image translation
- **Generator G:** Translates content image from domain X to stylized image in domain Y
- **Generator F:** Translates back from domain Y to domain X for cycle-consistency
- **Discriminator D_Y:** Distinguishes between real and generated images in domain Y
- **Cycle-Consistency Loss:** Ensures X → Y → X returns original image
- **Adversarial Loss:** Ensures generated images are indistinguishable from real images

### 4.2 System Components
1. **Image Preprocessing Module:** Resizes, normalizes, and augments images
2. **Generator Network (G and F):** Applies transformations between domains
3. **Discriminator Network (D_X and D_Y):** Validates if generated image is real or fake
4. **Training Engine:** Trains model using cycle-consistency and adversarial loss
5. **Inference Engine:** Generates stylized outputs from trained models
6. **Output Module:** Saves the stylized image

### 4.3 Functional Requirements

| ID | Requirement |
|----|-------------|
| FR01 | Accept content image and style image as input |
| FR02 | Preprocess input images (resize, normalize, augment) |
| FR03 | Use Generator G to transform content to style domain |
| FR04 | Use Generator F to transform style domain image back to content domain |
| FR05 | Compute adversarial loss using Discriminators D_X and D_Y |
| FR06 | Compute cycle-consistency loss for training stability |
| FR07 | Optimize generator and discriminator losses iteratively |
| FR08 | Generate stylized image from trained model |
| FR09 | Support real-time inference with acceptable latency |

### 4.4 Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR01 | Train with unpaired datasets of at least 1,000 images per domain |
| NFR02 | GPU acceleration for training time under 5 minutes per epoch |
| NFR03 | Support image uploads up to 1024×1024 resolution |
| NFR04 | Produce stylized output within 3 seconds during inference |
| NFR05 | Total model size not exceeding 500MB |
| NFR06 | Use less than 4GB VRAM during inference |
| NFR07 | Implemented using PyTorch 2.0 with easy extensibility |

---

## Chapter 5: Software Architecture and Low Level Design

### 5.1 Algorithms

#### Algorithm 1: CycleGAN Training for Style Transfer
```
Input: Source Domain X, Target Domain Y
Output: Generator G that maps X to Y (Stylized Images)

1. Initialize Generator G: X → Y and F: Y → X
2. Initialize Discriminators D_X and D_Y
3. For each training iteration:
   a. Sample batch x from X, y from Y
   b. Generate G(x) and F(y)
   c. Compute adversarial loss for G and F
   d. Compute cycle consistency loss: L_cyc = ||F(G(x)) - x|| + ||G(F(y)) - y||
   e. Compute total loss: L_total = L_GAN + λ · L_cyc
   f. Backpropagate and update G, F, D_X, D_Y
4. Return Generator G
```

### 5.2 System Architecture

#### Class Diagram Components
- **UserInterface:** uploadImage(), selectStyle(), showResult()
- **StyleTransferEngine:** loadModel(), preprocessImage(), applyStyle(), postprocessImage()
- **ResultDisplay:** displayStyledImage(), downloadImage()

#### Component Diagram
- **Frontend:** React/Streamlit GUI (Image Upload, Style Selection)
- **Backend:** Flask/PyTorch Server
  - CycleGAN Style Transfer Service
- **Database (Optional):** Stores uploaded images, metadata, logs

#### Deployment
- **Client:** Browser or App
- **Server:** Google Colab or AWS with GPU Runtime
- **Storage:** Cloud Storage Bucket for model weights and images

---

## Chapter 6: Conclusion

### 6.1 Summary
The project "AI-Based Neural Style Transfer" successfully implements CycleGAN for style transfer (unpaired image-to-image translation) across four distinct artistic styles.

**Implemented Style Models:**
1. **One Piece Style:** Transforms human faces into One Piece anime character style with bold outlines and vibrant colors
2. **Disney Style:** Transforms human faces into Disney animated character style with smooth gradients and expressive features
3. **Studio Ghibli Style:** Transforms human faces into Ghibli character style with soft watercolor textures
4. **Van Gogh Style:** Transforms landscape photographs into Van Gogh painting style with distinctive brush strokes

**Key Achievements:**
- Successfully created custom datasets through frame extraction from animation sources
- Stylized images preserve content features while adapting to target artistic styles
- Style-specific models capture unique characteristics of each animation/art style
- Modular architecture separates concerns effectively
- Cloud-based deployment with GPU acceleration ensures efficient performance
- Processing time optimized to under 3 seconds per image

**Dataset Creation Highlights:**
- Animation frames extracted from One Piece episodes, Disney movies, and Studio Ghibli films
- Human face datasets sourced from Kaggle for animation style training
- Van Gogh paintings and landscape photographs sourced from Kaggle
- Custom frame extraction algorithm developed using OpenCV and face detection

### 6.2 Scope for Future Work
1. **Multi-style Transfer:** Apply multiple styles to a single image or allow style blending
2. **Real-time Inference:** Optimize models for mobile or edge devices
3. **Increased Dataset Diversity:** Expand training dataset with more animation styles (Pixar, DreamWorks, etc.)
4. **User Personalization:** Allow users to train with their own artworks or custom animation styles
5. **AR/VR Integration:** View styled images in immersive environments
6. **Video Style Transfer:** Extend the system to process video content with temporal consistency
7. **Higher Resolution Output:** Support for higher resolution image generation (512×512 or 1024×1024)

---

## Appendices

### Appendix A: Project Planning
<!-- TODO: Add Gantt chart and budget estimation -->

### Appendix B: Dataset Details

#### B.1 One Piece Style Transfer Dataset

**Domain X: Human Faces**
| Attribute | Details |
|-----------|---------|
| Source | Kaggle Human Faces Dataset |
| Total Images | 3000+ |
| Image Format | JPG/PNG |
| Resolution | 256 × 256 pixels |
| Content Type | Real human face photographs |
| Diversity | Various ages, genders, ethnicities |

**Domain Y: One Piece Character Frames**
| Attribute | Details |
|-----------|---------|
| Source | One Piece anime episodes and movies |
| Extraction Method | Custom character detection algorithm using OpenCV |
| Total Images | 2500+ |
| Image Format | PNG |
| Resolution | 256 × 256 pixels |
| Content Type | One Piece character faces |
| Style Characteristics | Bold outlines, exaggerated expressions, vibrant colors |

**One Piece Frame Extraction Details:**
- Episodes from multiple story arcs were processed
- Character detection focused on main characters (Luffy, Zoro, Nami, etc.)
- Frames with clear frontal or semi-frontal character views were prioritized
- Quality filtering removed blurry or partially obscured frames

#### B.2 Disney Style Transfer Dataset

**Domain X: Human Faces**
| Attribute | Details |
|-----------|---------|
| Source | Kaggle Human Faces Dataset |
| Total Images | 3000+ |
| Image Format | JPG/PNG |
| Resolution | 256 × 256 pixels |
| Content Type | Real human face photographs |

**Domain Y: Disney Character Frames**
| Attribute | Details |
|-----------|---------|
| Source | Disney animated feature films |
| Extraction Method | Custom character detection algorithm |
| Total Images | 2000+ |
| Image Format | PNG |
| Resolution | 256 × 256 pixels |
| Content Type | Disney animated character faces |
| Style Characteristics | Smooth gradients, large expressive eyes, soft color palettes |

**Disney Frame Extraction Details:**
- Movies processed: Frozen, Tangled, Moana, Encanto, and others
- Focus on human and humanoid character faces
- High-quality frames with clear character expressions selected
- Frames with consistent lighting and composition prioritized

#### B.3 Studio Ghibli Style Transfer Dataset

**Domain X: Human Faces**
| Attribute | Details |
|-----------|---------|
| Source | Kaggle Human Faces Dataset |
| Total Images | 3000+ |
| Image Format | JPG/PNG |
| Resolution | 256 × 256 pixels |
| Content Type | Real human face photographs |

**Domain Y: Ghibli Character Frames**
| Attribute | Details |
|-----------|---------|
| Source | Studio Ghibli animated films |
| Extraction Method | Custom character detection algorithm |
| Total Images | 2000+ |
| Image Format | PNG |
| Resolution | 256 × 256 pixels |
| Content Type | Studio Ghibli character faces |
| Style Characteristics | Soft watercolor textures, naturalistic expressions, warm tones |

**Ghibli Frame Extraction Details:**
- Movies processed: Spirited Away, Howl's Moving Castle, My Neighbor Totoro, Princess Mononoke, Kiki's Delivery Service
- Focus on human character faces with Ghibli's distinctive art style
- Frames capturing the soft, hand-drawn aesthetic were prioritized
- Diverse character expressions and poses included

#### B.4 Van Gogh Style Transfer Dataset

**Domain X: Real-World Landscapes**
| Attribute | Details |
|-----------|---------|
| Source | Kaggle Landscape Photographs Dataset |
| Total Images | 2500+ |
| Image Format | JPG/PNG |
| Resolution | 256 × 256 pixels |
| Content Type | Natural scenery, outdoor scenes, landscapes |
| Categories | Mountains, fields, skies, villages, nature |

**Domain Y: Van Gogh Paintings**
| Attribute | Details |
|-----------|---------|
| Source | Kaggle Van Gogh Paintings Dataset |
| Total Images | 400+ |
| Image Format | JPG |
| Resolution | 256 × 256 pixels |
| Content Type | Original Van Gogh artworks (landscapes and portraits) |
| Style Characteristics | Swirling brush strokes, vibrant colors, post-impressionist, impasto technique |

**Van Gogh Dataset Composition:**
- Landscape paintings: Starry Night, Wheat Fields, Olive Trees series
- Portrait paintings: Self-portraits and figure studies
- Both landscape and painted subjects from Kaggle for comprehensive style learning

#### B.5 Frame Extraction Process (Animation Styles)

**Technical Implementation:**
1. **Video Loading:** Input video files loaded using OpenCV VideoCapture
2. **Frame Iteration:** Sequential frame-by-frame processing
3. **Face/Character Detection:** Pre-trained detection model (MTCNN or similar) applied
4. **Confidence Filtering:** Frames with detection confidence > 0.8 retained
5. **Bounding Box Extraction:** Detected face region cropped with padding
6. **Similarity Check:** Perceptual hashing used to avoid duplicate frames
7. **Quality Filtering:** Blurry or low-quality frames removed
8. **Resizing:** Final images resized to 256×256 pixels
9. **Storage:** Extracted frames saved in PNG format

**Frame Selection Criteria:**
- Clear visibility of character face
- Minimal motion blur
- Good lighting conditions
- Frontal or semi-frontal view preferred
- Diverse expressions and poses

#### B.6 Data Preprocessing Pipeline

**Step 1: Loading**
- Images loaded using PIL (Python Imaging Library) or OpenCV
- Color space converted to RGB if necessary

**Step 2: Resizing**
- All images resized to 256×256 pixels
- Aspect ratio preserved with center cropping when needed

**Step 3: Normalization**
- Pixel values normalized from [0, 255] to [-1, 1] range
- Formula: normalized = (pixel / 127.5) - 1

**Step 4: Data Augmentation (Training Only)**
- Random horizontal flip (50% probability)
- Random rotation (±10 degrees)
- Random brightness/contrast adjustment

**Step 5: Batching**
- Images organized into batches of size 4
- Shuffled randomly during training

#### B.7 Dataset Statistics Summary

| Style | Domain X Images | Domain Y Images | Total Training Pairs |
|-------|-----------------|-----------------|---------------------|
| One Piece | 3000+ | 2500+ | 5500+ |
| Disney | 3000+ | 2000+ | 5000+ |
| Studio Ghibli | 3000+ | 2000+ | 5000+ |
| Van Gogh | 2500+ | 400+ | 2900+ |

### Appendix C: Configuration Details
<!-- TODO: Add GitHub repository link -->

---

## References
1. Gatys, L.A., Ecker, A.S., Bethge, M. (2016). "Image Style Transfer Using CNNs" - CVPR
2. Johnson, J., Alahi, A., Fei-Fei, L. (2016). "Perceptual Losses for Real-Time Style Transfer" - ECCV
3. Huang, X., Belongie, S. (2017). "Arbitrary Style Transfer with AdaIN" - ICCV
4. Park, D.H., Berg, A.C., Berg, T.L. (2019). "Style-Attentional Networks" - CVPR
5. Kotovenko, D., Sanakoyeu, A., Ommer, B. (2019). "Content and Style Disentanglement" - ICCV
6. Li, Y. et al. (2017). "Universal Style Transfer via Feature Transforms" - NeurIPS
7. Zhang, H. et al. (2018). "Multi-Style Generative Network" - ECCV
8. Ruder, M., Dosovitskiy, A., Brox, T. (2016). "Artistic Style Transfer for Videos" - GCPR
9. Jing, Y. et al. (2020). "Neural Style Transfer: A Review" - IEEE TVCG
10. Zhang, Y., Dana, K. (2020). "Cross-Domain Correspondence Learning" - CVPR
