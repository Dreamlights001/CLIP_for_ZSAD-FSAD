# 对比语言-图像预训练方法重新定义零样本异常检测：综述与展望
语言：[英文](https://github.com/Dreamlights001/CLIP_for_ZSAD-FSAD),[中文](https://github.com/Dreamlights001/CLIP_for_ZSAD-FSAD/blob/main/README-zh.md)
## 1. 引言

工业瑕疵检测（IAD）属于计算机视觉在制造行业的一种典型应用，旨在自动识别工业生产过程中的异常产品。此任务通常包含三个层面的问题："什么是缺陷"（分类）、"缺陷在哪里"（定位）以及"有多少缺陷"（分割/计数）。

### 常见的三种工业瑕疵类型：

- **纹理异常**：指产品表面的局部缺陷，不改变其整体结构。例如划痕、污渍、变色、凹痕等。
- **结构异常**：指产品组件的形状、轮廓或完整性发生改变。例如部件缺失、断裂、变形或错位。
- **逻辑异常**：指更复杂的、依赖上下文的异常。单个组件可能完好无缺，但它们的组合或状态不符合预期逻辑。例如组件装配顺序错误、使用了错误的组件或物体处于异常状态（如开关在错误的位置）[1,2]。

### 传统机器学习/深度学习方法的局限

传统方法主要分为两类：

- **基于重建的方法**：以Unet、自动编码器（AE）、GAN等为代表，通过学习大量正常样本的特征来重建输入图像[1]。如果重建图像与原图差异过大，则被判定为异常。
- **基于嵌入的方法**：如PaDiM[3]，通过将正常样本的特征（通常来自ImageNet预训练模型）映射到一个紧凑的特征空间，并建立其分布，偏离此分布的样本被视为异常。

这些传统方法的局限性在于：它们严重依赖"所见即正常"的假设，需要大量、纯净的正常样本进行训练，并且难以泛化到训练中未见过的全新异常类型。

### 零样本/少样本范式的出现

最近随着CLIP[4]（对比语言-图像预训练）、分割一切（SAM）[5]等基础模型的问世，模型得以利用其强大的图文对齐能力，通过自然语言描述来理解"正常"与"异常"的概念，从而在无需任何目标域训练样本的情况下进行检测。利用预训练大模型如CLIP可有效应对样本稀缺问题，展现出解决零样本异常检测（ZSAD）和少样本异常检测（FSAD）的潜力。

因此我们聚焦CLIP模型的发展历程及其在工业瑕疵检测中的应用，简要对比其他预训练模型如SAM和DINO[6]，并与传统方法如UNet进行比较。同时总结工业瑕疵分类体系、零样本范式下的基准方法、改进流派，并分析技术路径的差异化演变。

## 2. 相关工作

在这一节我们会介绍CLIP模型在工业瑕疵检测方面催生出的新范式。

### 2.1 CLIP模型的基本架构与预训练方法

CLIP是由OpenAI在2021年提出的一种视觉-语言跨模态预训练模型，其核心思想是通过对比学习方法，在大规模图像-文本对上学习视觉与语言表征之间的对齐关系。

原始的CLIP模型通过联合训练图像编码器（如ViT或ResNet）和文本编码器，在4亿对图像-文本数据上进行对比学习，目标是最大化匹配图像与文本描述的相似性。这种设计使其能够将自然语言描述与视觉特征直接关联，例如通过提示词（如"正常螺丝的照片"）实现类别推理[7]。

在预训练过程中，模型通过最大化匹配图像-文本对的相似度，同时最小化不匹配对的相似度，学习到一个共享的嵌入空间，使得语义相关的图像和文本在该空间中距离更近。这种大规模预训练使得CLIP具备了强大的零样本迁移能力：对于未见过的视觉概念，只需提供相应的文本描述，模型即可进行识别和分类，而无需额外的训练样本。这一特性使CLIP特别适合于工业瑕疵检测场景，因为在该场景中，收集大量瑕疵样本进行模型训练通常十分困难。

### 2.2 CLIP在异常检测中的核心技术演进

CLIP的核心能力在于将图像和文本映射到同一语义空间。零样本瑕疵检测的基本思路是，计算图像（或图像块）的嵌入特征与"正常"和"异常"文本提示的嵌入特征之间的相似度。这种方式通过图像-文本对比学习实现了跨模态语义对齐，具备很强的跨领域自适应能力，甚至能够在未见过的目标域上进行异常检测及分割，这一特性使其在工业瑕疵检测中展现出独特优势。

CLIP的发展经历了从基础架构到工业适配的演进过程，在工业应用场景中，CLIP的核心优势体现在三个方面：零样本/少样本学习能力、语义匹配与提示学习以及多尺度特征提取。

WinCLIP[8]是CLIP应用于零样本异常分割的开创性工作之一。它不直接处理整张图像，而是采用滑动窗口策略：将图像划分为不同尺度的窗口；计算每个窗口的图像特征与一组组合式提示的文本特征（如"正常状态" vs "异常状态"）之间的CLIP相似度得分；聚合多尺度的得分图，生成最终的异常热力图，从而实现像素级定位。

### 2.3 零样本范式下的其他基准方法对比

CLIP、SAM和DINO系列作为三类主流预训练模型，在工业缺陷检测中展现出不同的优势和局限性。

- **SAM**：其优势在于像素级定位精度实现零样本分割。它擅长根据提示（点、框、掩码）分割出"万物"，但它本身不知道什么是异常，需要依赖精准的提示词，且在复杂工业场景中易受背景干扰。

- **DINO**：是一个自监督学习模型，它不使用文本，仅通过视觉信息（如视图的一致性）直接从图像生成学习信号，无需标注数据。DINO（特别是DINOv2[9]）学习到的视觉特征具有极强的细粒度分辨能力和空间对应关系，对纹理和结构的细微变化非常敏感，其通常作为强大的特征提取器。Grounding DINO[10]使用任意文本作为查询实现了令人鼓舞的开放集对象检测能力；DINOv3[11]在密集预测任务中表现优异，如SPADE-ViT[12]模型在MVTec[13]数据集上达到94.0%的平均PRO精度。

基于CLIP、SAM和DINO等基础模型的互补特性，研究者提出了多种多模型融合策略，以实现更全面、准确的工业瑕疵检测。如ClipSAM[14]，CLIP负责"识别"异常的语义，SAM负责"分割"异常的边界；SAA+则采取Grounding DINO模型实现的异常区域生成器粗略地检索粗异常区域，再采用SAM生成逐像素的高质量掩码，通过混合提示正则化实现了无需训练的异常分割。

### 2.4 基于CLIP的零样本/少样本异常检测方法改进

尽管WinCLIP证实了CLIP的有效性，但也暴露了其局限性，例如对物体本身的语义（如"瓶子"）的关注超过了对异常状态（如"破损"）的关注。后续研究围绕此问题形成了三大改进流派：

1. **基于提示学习的改进**：通过优化文本提示的表示，使模型更好地理解工业场景中的正常和异常概念，通过动态生成文本提示，提升对齐。这种方式聚焦语义泛化，但对噪声敏感。

2. **引入适配器的改进**：通过在CLIP模型中引入轻量级的适配器模块，调整模型的视觉或文本表征，使其更适合工业异常检测任务。这些适配器通常使用少量工业数据进行微调，在保持CLIP原有知识的同时，增强其对工业异常的敏感度。这种方式减少了训练的参数量，降低计算成本，适合边缘部署。

3. **基于模型组装的改进**：此流派倾向于融合不同基础模型的优势，特别是将CLIP的语义理解能力与SAM等强大的分割能力相结合。这种方式提升精度，但增加了模型复杂度。

### 2.5 后续源于这三种流派的改进方法

#### 基于提示学习的改进

在提示学习方向，研究核心从静态模板转向动态与上下文感知的提示生成。

- 早期工作如AnomalyCLIP[15]提出了Object-agnostic Prompt Learning，使模型摆脱对特定物体类别的依赖。
- AdaCLIP[16]采用了混合可学习提示，结合了静态通用提示与动态图像特定提示。
- VCP-CLIP[17]进一步提出了视觉上下文提示模型，利用图像自身的全局和局部特征来动态生成文本提示。
- GlocalCLIP[18]则集成了对象无关的全局-局部提示学习机制。
- 最新的研究如CoPS[19]实现了条件提示合成框架，通过视觉特征显式合成动态提示。
- Crane[20]结合了上下文引导的提示学习与注意力精炼，有效解决了CLIP的空间错位问题。
- AFR-CLIP[21]则通过图像引导的文本校正，实现了从"无状态"到"有状态"的异常特征矫正，在选用的四个主流工业数据集上均达到了最高的图像级和像素级AUROC。

#### 基于适配器集成的改进

在模型适配方向，焦点在于以最小参数代价实现CLIP的领域特化。

- AF-CLIP[22]进行了异常聚焦的CLIP适配，通过轻量级适配器同步优化类级与补丁级特征，并引入了多尺度空间聚合机制。
- AA-CLIP[23]提出了异常感知CLIP，通过构建"异常感知文本锚点"并从文本、视觉双维度补全CLIP的异常感知能力，解决了"异常无意识现象"。
- PA-CLIP[24]则利用了伪异常感知，通过构建区分背景信息与真实异常的内存库来降低误检。
- 值得关注的还有AdaptCLIP[25]，它仅在CLIP的输入/输出端添加了视觉、文本和提示-查询三个轻量适配器，实现了高效的零样本跨域泛化。

研究表明，此类适配器方法性能卓越，例如IAD-CLIP[26]框架在MVTec AD数据集上实现了92.1%的图像级AUROC以及94.6%的AUPR，充分展示了CLIP在零样本工业异常检测中的潜力。

#### 基于模型组装的改进

在模型组装方向，旨在通过模型协同弥补单一模型的能力短板。

- ClipSAM[14]探索了CLIP与SAM模型的协作，利用CLIP进行语义级异常判断，并借助SAM实现像素级精密分割。
- MultiADS[27]虽然在架构上可被视为一种组装，其核心贡献在于首个实现了零样本下的多类型异常检测与分割，能够为不同缺陷类型生成特定异常掩码，并区分产品中同时存在的多种缺陷。
- 此外，FiLo[28]结合Grounding DINO和CLIP作为基本框架，专注于通过细粒度描述和高质量定位进行检测，也体现了组装模型以兼顾不同粒度分析的思想，其改进版FiLo++[29]则使用了融合的细粒度描述和可变形定位。

#### 其他先进策略

除了这三类常见的方式，研究者提出了多种改进策略：

- CLIP Surgery[30]通过移除冗余特征和修改注意力机制增强解释性。
- LECLIP[31]框架引入局部对齐模块和回声注意力，在15个工业与医学数据集上实现最优性能。
- MGFD-CLIP则通过解耦全局与局部特征，在零样本任务中图像级AP相对提升9.21%。
- MissingClip[32]针对模态缺失问题（如点云数据延迟），通过双注意力机制和混合语义提示重构缺失模态特征。

## 3. 结论与讨论

基于CLIP的零样本工业异常检测范式，成功地利用了视觉语言模型强大的图文对齐能力和泛化能力，有效解决了工业领域异常样本稀缺、标注成本高昂的挑战。该领域的发展已清晰的形成提示学习、模型适配、模型组装三大改进路径。

总体而言，CLIP极大地拓宽了工业异常检测的应用边界，推动了"训练成本低、泛化能力强"的新型检测系统的发展。基于CLIP的零样本工业瑕疵检测已成为一个充满活力的研究领域。其发展路径清晰地展现了从利用先验，到优化先验，再到融合先验的演进趋势。

## 4. 讨论与未来方向

尽管CLIP在ZSAD/FSAD任务中取得了显著成就，但在实际工业应用中仍面临以下挑战，这也构成了未来对这些挑战的主要研究方向：

- **领域差距与语义漂移**：需要更高效、领域特异性的适配策略来桥接CLIP预训练数据与工业图像的领域鸿沟，例如在特征空间中进行校准或正则化。

- **局部细节感知不足**：CLIP的注意力机制偏向全局语义，对细微或复杂的旋转缺陷等局部异常特征感知力有限，需要增强对补丁级或多尺度特征的捕捉能力。

- **一致性异常问题**：当测试集中出现重复的、相似的缺陷模式时，基于相似度计算的方法性能会严重下降，需要识别并过滤这些异常。

- **可解释性与推理能力**：现有方法多输出异常分数或热力图，缺乏自然语言的解释和推理过程。结合多模态大语言模型以实现多轮人机对话和一致性推理，如IAD-R1[33]旨在强化一致性推理。

- **多模态与泛化性**：将异常检测任务扩展到更复杂的场景，如3D点云异常检测或跨域泛化。

未来的研究将趋向于语义、空间、推理三维度的协同增强：利用更精细的文本语义指导（提示学习），通过更灵活的架构或适配器实现高精度定位，并最终融入大型基础模型的推理能力，以构建一个可解释、高精度、高泛化性的零样本工业检测系统。

## 参考文献

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
