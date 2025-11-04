# Contrastive Language-Image Pre-Training Methods Redefining Zero-shot/Few-shot Anomaly Detection: A Survey and Outlook
Language: English[],Chinese[]
## 1. Introduction

Industrial Anomaly Detection (IAD) is a typical application of computer vision in the manufacturing industry, aiming to automatically identify anomalous products during the production process. This task generally involves three levels of questions: "What is the defect" (Classification), "Where is the defect" (Localization), and "How many defects are there" (Segmentation/Counting).

### Three Common Types of Industrial Anomalies:

1. **Texture Anomalies**: Refer to localized defects on the product surface that do not alter its overall structure. Examples include scratches, stains, discoloration, dents, etc.
2. **Structural Anomalies**: Refer to changes in the shape, contour, or integrity of product components. Examples include missing parts, fractures, deformation, or misalignment.
3. **Logical Anomalies**: Refer to more complex, context-dependent anomalies. An individual component may be intact, but their combination or status does not meet the expected logic. Examples include incorrect assembly sequence, use of wrong components, or an object being in an abnormal state (e.g., a switch in the wrong position)[1,2].

### Limitations of Traditional Machine Learning/Deep Learning Methods

Traditional methods are mainly divided into two categories:

- **Reconstruction-based Methods**: Represented by Unet, Autoencoders, and GANs, which reconstruct the input image by learning features from a large number of normal samples[1]. If the reconstructed image differs significantly from the original image, it is identified as anomalous.
- **Embedding-based Methods**: Such as PaDiM[3] via mapping the features of normal samples (typically derived from ImageNet pre-trained models) into a compact feature space and establishing their distribution, samples deviating from this distribution are considered anomalous.

The limitations of these traditional methods are that they heavily rely on the assumption of "what is seen is normal," require a large number of clean normal samples for training, and struggle to generalize to entirely new anomaly types unseen during training.

### Zero-Shot/Few-Shot Paradigms to Address This Challenge

Recently, with the advent of foundational models like Contrastive Language-Image Pre-Training (CLIP[4]) and Segment Anything Model (SAM[5]), models can leverage their powerful vision-language alignment capabilities to understand the concepts of "normal" and "anomalous" through natural language descriptions, thereby performing detection without the need for any target-domain training samples. The use of large pre-trained models such as CLIP is highly effective in tackling the problem of sample scarcity, showing potential for solving zero-shot anomaly detection (ZSAD) and few-shot anomaly detection (FSAD).

Therefore, we focus on the development trajectory of the CLIP model and its application in industrial anomaly detection. We will briefly compare CLIP with other pre-trained models, such as SAM and Distillation with No Labels (DINO[6]), and contrast them with traditional methods like UNet. Simultaneously, this survey will summarize the industrial anomaly classification system, the benchmark methods within the zero-shot paradigm, the main streams of improvement, and analyze the differentiated evolution of technical pathways.

## 2. Related Works

In this section, we will introduce the new paradigm catalyzed by the CLIP model in the field of industrial anomaly detection.

### 2.1 The Basic Architecture and Pre-training Method of the CLIP Model

CLIP is a vision-language cross-modal pre-training model proposed by OpenAI in 2021. Its core idea is to learn the alignment relationship between visual and language representations over a massive dataset of image-text pairs using a contrastive learning approach.

The original CLIP model jointly trains an image encoder (such as ViT or ResNet) and a text encoder using contrastive learning on 400 million image-text data pairs. The goal is to maximize the similarity between matching image and text description. This design enables it to directly associate natural language descriptions with visual features, for instance, by using prompts (such as "a photo of a normal screw") to perform category reasoning[7].

During the pre-training process, the model learns a shared embedding space by maximizing the similarity of matching image-text pairs while minimizing the similarity of non-matching pairs. This results in semantically related images and texts being closer in this space. This large-scale pre-training grants CLIP powerful zero-shot transfer capability: for unseen visual concepts, the model can perform recognition and classification simply by providing the corresponding text description, without requiring additional training samples. This characteristic makes CLIP particularly suitable for industrial anomaly detection scenarios, where collecting a large number of anomaly samples for model training is often extremely difficult.

### 2.2 Core Technical Evolution of CLIP in Anomaly Detection

CLIP's core capability lies in mapping images and text into a shared semantic space. The fundamental idea of zero-shot anomaly detection is to compute the similarity between the embedded features of the image (or image patches) and the embedded features of "normal" and "anomalous" Text Prompts. This approach achieves cross-modal semantic alignment through image-text contrastive learning, possessing strong cross-domain adaptation capability. It can even perform anomaly detection and segmentation on unseen target domains, a feature that gives it a unique advantage in industrial anomaly detection.

The development of CLIP has undergone an evolution from its basic architecture to industrial adaptation. In industrial application scenarios, CLIP's core advantages are manifested in three aspects: Zero-Shot/Few-Shot Learning Capability, Semantic Matching and Prompt Learning and Multi-Scale Feature Extraction.

WinCLIP[8] is one of the pioneering works applying CLIP to zero-shot anomaly segmentation. Instead of processing the entire image directly, it adopts a Window-based strategy: dividing the image into windows (Patches) of different scales; computing the CLIP similarity score between the image features of each window and the text features of a set of Compositional Prompts, such as "normal state" and "anomalous state"); and finally aggregating the multi-scale score maps to generate the final anomaly heatmap, thereby achieving pixel-level localization.

### 2.3 Comparison with Other Benchmark Methods under Zero-Shot Paradigm

As three mainstream pre-trained models, CLIP, SAM, and the DINO series exhibit distinct advantages and limitations in industrial defect detection.

- **SAM (Segment Anything Model)**: Its advantage lies in achieving zero-shot segmentation with pixel-level localization accuracy. It excels at segmenting "everything" based on prompts (points, boxes, masks), but it intrinsically does not know what an anomaly is. It requires reliance on precise prompts and is susceptible to background interference in complex industrial scenes.

- **DINO (Distillation with No Labels)**: As a Self-Supervised Learning (SSL) model, it does not use text but generates learning signals directly from visual information (such as view consistency) without the need for labeled data. DINO (especially DINOv2[9]) learns visual features with extremely strong fine-grained discrimination and spatial correspondence, making it highly sensitive to subtle textural and structural changes. It is commonly used as a powerful feature extractor. Grounding DINO[10] achieved encouraging open-set object detection capability using arbitrary text as a query, while DINOv3[11] performs excellently in dense prediction tasks. For example, the SPADE-ViT[12] model achieved a 94.0% average PRO score on the MVTec[13] dataset.

Based on the complementary nature of foundational models like CLIP, SAM, and DINO, researchers have proposed various multi-model fusion strategies to achieve more comprehensive and accurate industrial anomaly detection. For instance, ClipSAM[14] is a collaboration between CLIP and SAM: CLIP is responsible for the semantic "identification" of the anomaly, while SAM is responsible for the "segmentation" of the anomaly boundary. Similarly, SAA+ employs an anomaly region generator implemented by the Grounding DINO model to coarsely retrieve abnormal areas, and then utilizes SAM to generate high-quality pixel-wise masks. This framework achieves training-free anomaly segmentation through Hybrid Prompt Regularization.

### 2.4 Improvements on CLIP-based Zero-Shot/Few-Shot Anomaly Detection Methods

Although WinCLIP validated the effectiveness of CLIP, it also exposed its limitations, such as its tendency to focus on the semantics of the object itself (e.g., "bottle") rather than the state of the anomaly (e.g., "broken"). Subsequent research has centered on this issue, leading to the formation of three major streams of improvement (a large number of works in the references can be categorized here):

1. **Prompt Learning-Based Improvements**: These methods optimize the representation of text prompts to enable the model to better understand the concepts of "normal" and "anomalous" in industrial scenarios, often by dynamically generating text prompts to enhance alignment. This approach focuses on semantic generalization but is sensitive to noise.

2. **Adapter-Based Improvements**: This stream introduces lightweight adapter modules into the CLIP model to adjust its visual or textual representations, making them more suitable for the industrial anomaly detection task. These adapters are typically fine-tuned with a small amount of industrial data to enhance the model's sensitivity to industrial anomalies while preserving CLIP's pre-trained knowledge. This approach reduces the number of training parameters and lowers computational costs, making it suitable for edge deployment.

3. **Model Assembly-Based Improvements**: This stream tends to fuse the strengths of different foundational models, specifically by combining CLIP's semantic understanding capability with the powerful segmentation capabilities of models like SAM (Segment Anything Model). This approach improves accuracy but increases model complexity.

### 2.5 Methods Later Originate from These Three Streams of Improvement

#### Improvements Based on Prompt Learning

In the prompt learning direction, the research focus has shifted from static templates to dynamic and context-aware prompt generation.

- Early work, such as AnomalyCLIP[15], proposed Object-agnostic Prompt Learning, enabling the model to break free from dependence on specific object categories.
- AdaCLIP[16] employed Hybrid Learnable Prompts, combining static general prompts with dynamic image-specific prompts.
- VCP-CLIP[17] further proposed the Visual Context Prompting Model, which utilizes the image's own global and local features to dynamically generate text prompts.
- GlocalCLIP[18] integrated an object-agnostic global-local prompt learning mechanism.
- Latest research, such as CoPS[19], implements a Conditional Prompt Synthesis framework that explicitly synthesizes dynamic prompts from visual features.
- Crane[20] combines Context-Guided Prompt Learning with Attention Refinement to effectively solve CLIP's spatial misalignment problem.
- AFR-CLIP[21] achieves a "Stateless-to-Stateful" Anomaly Feature Rectification through image-guided text calibration, achieving the highest image-level and pixel-level AUROC across four selected mainstream industrial datasets.

#### Improvements Based on Adapter Integration

In the model adaptation direction, the focus is on achieving domain specialization of CLIP with minimal parameter cost.

- AF-CLIP[22] performed Anomaly-Focused CLIP Adaptation, which synchronously optimizes class-level and patch-level features via lightweight adapters and introduces a multi-scale spatial aggregation mechanism.
- AA-CLIP[23] proposed Anomaly-Aware CLIP, which solves the "Anomaly-Unawareness Phenomenon" by constructing "Anomaly-Aware Text Anchors" and complementing CLIP's anomaly perception from both visual and textual dimensions.
- PA-CLIP[24] utilized Pseudo-Anomaly Awareness, reducing false positives by constructing a memory bank that distinguishes between background information and true anomalies.
- Also noteworthy is AdaptCLIP[25], which achieved efficient zero-shot cross-domain generalization by only adding three lightweight adapters—visual, textual, and prompt-query—to the input/output of CLIP.

Studies indicate that these adapter methods perform excellently; for instance, the IAD-CLIP[26] framework achieved 92.1% image-level AUROC and 94.6% AUPR on the MVTec AD dataset, fully demonstrating CLIP's potential in ZSAD and FSAD.

#### Improvements Based on Model Assembly

The Model Assembly approach aims to compensate for the shortcomings of a single model through model collaboration.

- ClipSAM[14] explored the Collaboration of CLIP and SAM models, utilizing CLIP for semantic-level anomaly judgment and leveraging SAM for pixel-level precise segmentation.
- MultiADS[27], while architecturally viewable as an assembly, its core contribution lies in being the first to achieve multi-type anomaly detection and segmentation in a zero-shot setting, capable of generating specific anomaly masks for different defect types and distinguishing multiple co-occurring defects on a product.
- Furthermore, FiLo[28] combined Grounding DINO and CLIP as its basic framework and focused on detection via Fine-Grained Description and High-Quality Localization, also embodying the assembly idea to account for different granularity analyses. Its improved version, FiLo++[29], used Fused Fine-Grained Descriptions and Deformable Localization.

#### Other Advanced Strategies

Beyond these three common streams, researchers have proposed various other improvement strategies. For example:

- CLIP Surgery[30] enhances explainability by removing redundant features and modifying the attention mechanism.
- LECLIP[31] introduces a Local Alignment module and Echo Attention within its framework, achieving optimal performance across 15 industrial and medical datasets.
- MGFD-CLIP decouples global and local features, achieving a 9.21% relative improvement in image-level AP in zero-shot tasks.
- MissingClip[32] addresses the problem of modality missing (such as delayed point cloud data) by reconstructing missing modality features through a dual-attention mechanism and hybrid semantic prompts.

## 3. Conclusion

The CLIP-based Zero-Shot Industrial Anomaly Detection paradigm has successfully leveraged the powerful vision-language alignment and generalization capabilities of Vision-Language Models (VLMs), effectively addressing the challenges of scarce anomaly samples and high annotation costs in the industrial domain. The development in this field has clearly organized into three major streams of improvement: Prompt Learning, Model Adaptation, and Model Assembly.

Overall, CLIP has significantly broadened the application boundaries of industrial anomaly detection, driving the development of new detection systems characterized by "low training cost and strong generalization capability." CLIP-based zero-shot industrial anomaly detection has emerged as a vibrant research area. Its evolutionary path clearly demonstrates a trend moving from utilizing priors (knowledge), to optimizing priors, and finally to fusing priors.

## 4. Discussions

Although CLIP has achieved significant success in ZSAD/FSAD tasks, it still faces the following challenges in practical industrial applications, which constitute the main research directions for the future:

- **Domain Gap and Semantic Drift**: There is a need for more efficient and domain-specific adaptation strategies to bridge the gap between CLIP's pre-training data and industrial images. This often involves techniques such as calibration or regularization in the feature space.

- **Insufficient Local Detail Perception**: CLIP's attention mechanism tends toward global semantics, leading to limited perceptual capability for subtle or complex local anomaly features, such as rotational defects. This requires enhancing the model's ability to capture patch-level or multi-scale features.

- **Consistent Anomaly Problem**: When repeated, similar defect patterns appear in the test set, the performance of similarity-based methods significantly degrades. There is a need to identify and filter out these anomalies.

- **Explainability and Reasoning Capability**: Existing methods mostly output anomaly scores or heatmaps but lack natural language explanations and reasoning processes. The integration of Multimodal Large Language Models (MLLMs) is necessary to achieve multi-turn human-machine dialogue and consistent reasoning, such as IAD-R1[33], which aims to reinforce consistent reasoning.

- **Multi-modality and Generalization**: IAD task needs to be extended to more complex scenarios, such as 3D point cloud anomaly detection or cross-domain generalization.

Future research will trend towards the synergistic enhancement across three dimensions: semantics, space, and reasoning: utilizing more refined text semantic guidance (Prompt Learning), achieving high-precision localization through more flexible architectures or adapters (Adapter Integration), and ultimately integrating the reasoning capabilities of large foundation models, such as MLLMs, to build an interpretable, high-accuracy, and highly generalizable zero-shot industrial inspection system.

## References

1. Xie, Guoyang, et al. "Im-iad: Industrial image anomaly detection benchmark in manufacturing." *IEEE Transactions on Cybernetics* 54.5 (2024): 2720-2733.
2. Zhao, Ying. "LogicAL: Towards logical anomaly synthesis for unsupervised anomaly localization." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2024.
3. Defard, Thomas, et al. "Padim: a patch distribution modeling framework for anomaly detection and localization." *International conference on pattern recognition*. Cham: Springer International Publishing, 2021.
4. Radford, Alec, et al. "Learning transferable visual models from natural language supervision." *International conference on machine learning*. PmLR, 2021.
5. Kirillov, Alexander, et al. "Segment anything." *Proceedings of the IEEE/CVF international conference on computer vision*. 2023.
6. Zhang, Hao, et al. "Dino: Detr with improved denoising anchor boxes for end-to-end object detection." *arXiv preprint arXiv:2203.03605* (2022).
7. Deng, Hanqiu, et al. "Anovl: Adapting vision-language models for unified zero-shot anomaly localization." *arXiv preprint arXiv:2308.15939* 2.5 (2023).
8. Jeong, Jongheon, et al. "Winclip: Zero-/few-shot anomaly classification and segmentation." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2023.
9. Oquab, Maxime, et al. "Dinov2: Learning robust visual features without supervision." *arXiv preprint arXiv:2304.07193* (2023).
10. Liu, Shilong, et al. "Grounding dino: Marrying dino with grounded pre-training for open-set object detection." *European conference on computer vision*. Cham: Springer Nature Switzerland, 2024.
11. Siméoni, Oriane, et al. "Dinov3." *arXiv preprint arXiv:2508.10104* (2025).
12. Park, Taesung, et al. "Semantic image synthesis with spatially-adaptive normalization." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2019.
13. Bergmann, Paul, et al. "MVTec AD—A comprehensive real-world dataset for unsupervised anomaly detection." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2019.
14. Li, Shengze, et al. "ClipSAM: CLIP and SAM collaboration for zero-shot anomaly segmentation." *Neurocomputing* 618 (2025): 129122.
15. Zhou, Qihang, et al. "Anomalyclip: Object-agnostic prompt learning for zero-shot anomaly detection." arXiv preprint arXiv:2310.18961 (2023).
16. Cao, Yunkang, et al. "Adaclip: Adapting clip with hybrid learnable prompts for zero-shot anomaly detection." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.
17. Qu, Zhen, et al. "Vcp-clip: A visual context prompting model for zero-shot anomaly segmentation." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.
18. Ham, Jiyul, Yonggon Jung, and Jun-Geol Baek. "GlocalCLIP: Object-agnostic Global-Local Prompt Learning for Zero-shot Anomaly Detection." arXiv preprint arXiv:2411.06071 (2024).
19. Chen, Qiyu, et al. "CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection." arXiv preprint arXiv:2508.03447 (2025).
20. Salehi, Alireza, et al. "Crane: Context-Guided Prompt Learning and Attention Refinement for Zero-Shot Anomaly Detections." *arXiv preprint arXiv:2504.11055* (2025).
21. Jingyi Yuan, et al. "Afr-clip: Enhancing zero-shot industrial anomaly detection with stateless-to-stateful anomaly feature rectification." arXiv preprint arXiv:2503.12910 (2025).
22. Fang, Qingqing, Wenxi Lv, and Qinliang Su. "AF-CLIP: Zero-Shot Anomaly Detection via Anomaly-Focused CLIP Adaptation." arXiv preprint arXiv:2507.19949 (2025).
23. Ma, Wenxin, et al. "Aa-clip: Enhancing zero-shot anomaly detection via anomaly-aware clip." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.
24. Pan, Yurui, et al. "PA-CLIP: Enhancing Zero-Shot Anomaly Detection through Pseudo-Anomaly Awareness." arXiv preprint arXiv:2503.01292 (2025).
25. Gao, Bin-Bin, et al. "AdaptCLIP: Adapting CLIP for Universal Visual Anomaly Detection." arXiv preprint arXiv:2505.09926 (2025).
26. Li, Zhuo, et al. "Iad-clip: Vision-language models for zero-shot industrial anomaly detection." 2024 International Conference on Advanced Mechatronic Systems (ICAMechS). IEEE, 2024.
27. Sadikaj, Ylli, et al. "MultiADS: Defect-aware Supervision for Multi-type Anomaly Detection and Segmentation in Zero-Shot Learning." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.
28. Gu, Zhaopeng, et al. "Filo: Zero-shot anomaly detection by fine-grained description and high-quality localization." Proceedings of the 32nd ACM International Conference on Multimedia. 2024.
29. Gu, Zhaopeng, et al. "FiLo++: Zero-/Few-Shot Anomaly Detection by Fused Fine-Grained Descriptions and Deformable Localization." arXiv preprint arXiv:2501.10067 (2025).
30. Li, Yi, et al. "Clip surgery for better explainability with enhancement in open-vocabulary tasks." arXiv e-prints (2023): arXiv-2304.
31. Liu, Yuyao, et al. "LECLIP: Boosting Zero-Shot Anomaly Detection with Local Enhanced CLIP." IEEE Transactions on Instrumentation and Measurement (2025).
32. Xu, Tianyi, et al. "MissingClip: An Industrial Anomaly Detection Method Under Modality Missing." International Conference on Wireless Artificial Intelligent Computing Systems and Applications. Singapore: Springer Nature Singapore, 2025.
33. Li, Yanhui, et al. "IAD-R1: Reinforcing Consistent Reasoning in Industrial Anomaly Detection." arXiv preprint arXiv:2508.09178 (2025).
