# DTLA-Net
# abstract
Medical image segmentation plays a crucial role in clinical diagnosis and treatment. Its accuracy and efficiency are vital for diagnosing, treating, and monitoring diseases. To address the challenges posed by high-resolution and complex medical images, where traditional Transformers may struggle to fully extract diverse global features, we propose a Direct2D Transformer with Linear Angle Attention network for multi-organ medical image segmentation (DTLA-Net). This innovative structure retains the UNet structure. To overcome the computational complexity of combining traditional convolutional neural networks (CNN) and Transformers, we introduce the Direct2D Transformer (D2D-Former) structure. It integrates the advantages of CNN to more effectively extract and process complex features in medical images within the encoding section of the network.  Additionally, given that medical images often contain noise and irregular shapes, we incorporate Linear Angle Attention (LA-Attention). This can more effectively capture and utilize the rich detail features in medical images, performing more stably and reliably in regions with complex structures and subtle local variations. Extensive experiments on the Synapse and ACDC public datasets demonstrate that DTLA-Net achieves significant improvements in medical image segmentation tasks, showcasing its potential and effectiveness in handling complex and high-resolution images.
# Usage
1. Download Google pre-trained ViT models

2. Prepare data (All data are available!)
 All data are available so no need to send emails for data.
3. Environment
   Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.
