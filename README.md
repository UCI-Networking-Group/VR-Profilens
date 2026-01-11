# VR-Profilens

**Abstract**
>> Virtual reality (VR) platforms and apps collect user sensor data, including motion, facial, eye, and hand data, in abstracted form. These data may expose users to unique privacy risks without their knowledge or meaningful awareness, yet the extent of these risks remains understudied. To address this gap, we propose VR ProfiLens, a framework to study user profiling based on VR sensor data and the resulting privacy risks across consumer VR apps. To systematically study this problem, we first develop a taxonomy rooted in the CCPA definition of personal information and expand it by sensor, app, and threat contexts to identify user attributes at risk. Then, we conduct a user study in which we collect VR sensor data from four sensor groups from real users interacting with 10 popular consumer VR apps, followed by a survey. We design and apply an analysis pipeline to demonstrate the feasibility of inferring user attributes using these data. Our results demonstrate the feasibility of user attribute inference, including sensitive personal information, with a moderately high to high risk (up to ∼90% F1 score) of being inferred from the sensor data. Through feature analysis, we further identify correlations among apps and sensors in inferring user attributes. Our findings highlight risks to users, including privacy loss, tracking, targeted advertising, and safety threats. Finally, we discuss both design implications and regulatory recommendations to enhance transparency and better protect users’ privacy in VR.

**Overview**
This repository contains the implementation of VR ProfiLens, a framework for analyzing user profiling risks in 10 consumer Virtual Reality apps.

**Citation**

Please cite our paper as follows:

```bibtex
@inproceedings{jarin2025VRProfilens,
  title     = {VR ProfiLens: User Profiling Risks in Consumer Virtual Reality Apps},
  author    = {Jarin, Ismat and Figueira, Olivia and and Duan, Yu and Le, Tu and Markopoulou, Athina},
  booktitle = {Symposium on Usable Security and Privacy (USEC)},
  volume    = {2026},
  year      = {2026}
}
```


