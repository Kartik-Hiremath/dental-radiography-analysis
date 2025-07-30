# Dental Radiography Analysis with PyTorch

## 📖 Overview

This project focuses on detecting common dental issues from radiography images using deep learning. The primary goal is to build and iteratively improve an object detection model capable of identifying **cavities, fillings, impacted teeth, and implants**.

The project is structured as a series of experiments, starting with a baseline model and progressively introducing advanced techniques to enhance performance. Two different backbone architectures, **Faster R-CNN with ResNet50** and **Faster R-CNN with MobileNetV3**, are trained and evaluated in parallel to compare their effectiveness.

---

## 📊 Dataset

The project utilizes the [Dental Radiography Dataset](https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography/data) from Kaggle. This dataset contains a collection of dental X-ray images with corresponding annotations for various dental conditions. The annotations are provided in PASCAL VOC format, which includes bounding boxes and class labels for each object of interest.

**Classes:**
1.  Cavity
2.  Fillings
3.  Impacted Tooth
4.  Implant

---

## 💡 Computer Vision Techniques

Before training the deep learning models, a series of computer vision techniques were applied to preprocess and enhance the quality of the dental radiographs.

* **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Dental X-ray images can often have poor contrast, making it difficult to distinguish between different tissues and dental conditions. CLAHE is an advanced histogram equalization technique that enhances the contrast of the images by operating on small regions of the image, called tiles, rather than the entire image. This helps to bring out details in the images that might otherwise be hidden.
* **Data Augmentation:** To improve the generalization of the models and prevent overfitting, a series of data augmentation techniques were applied to the training images. These techniques include:
    * **Horizontal Flipping:** The images were randomly flipped horizontally to create new training samples.
    * **Random Brightness and Contrast:** The brightness and contrast of the images were randomly adjusted to make the models more robust to changes in lighting conditions.
    * **Gaussian Noise:** Gaussian noise was added to the images to make the models more robust to noise.
    * **Shift, Scale, and Rotate:** The images were randomly shifted, scaled, and rotated to make the models more robust to changes in the position, size, and orientation of the objects.

---

## 🚀 Project Evolution & Methodology

The notebook is organized into five distinct parts, each building upon the last to demonstrate a systematic approach to model improvement.

### Part 1: Initial Model Training

* **Goal:** Establish a baseline performance.
* **Process:**
    * Two Faster R-CNN models were initialized with pre-trained COCO weights: one with a **ResNet50** backbone and another with a **MobileNetV3** backbone.
    * The models were trained on the dental radiography dataset using a standard training loop.
    * A custom weighted loss function (`fastrcnn_loss_weighted`) was implemented to apply higher penalties for misclassifying under-represented classes like 'Cavity'.
* **Outcome:** This initial training provided a baseline against which all subsequent improvements were measured. It highlighted the initial challenges, such as class imbalance and the models' tendencies to miss certain classes.

### Part 2: Weighted Sampling

* **Goal:** Address the class imbalance problem more directly at the data loading stage.
* **Process:**
    * Instead of only weighting the loss, we introduced `WeightedRandomSampler` from PyTorch.
    * We calculated weights for each sample in the training set based on the presence of rare classes ('Cavity' and 'Impacted Tooth').
    * The `DataLoader` used this sampler to oversample images containing these rare classes, ensuring the model saw them more frequently during training.
* **Outcome:** This technique helped the model learn the features of under-represented classes more effectively, leading to better recall for those classes.

### Part 3: Focal Loss

* **Goal:** Improve training by focusing on hard-to-classify examples.
* **Process:**
    * The standard cross-entropy loss function was replaced with **Focal Loss**.
    * Focal Loss dynamically down-weights the loss assigned to well-classified examples (easy examples) and focuses the model's attention on misclassified examples (hard examples).
    * This was implemented by patching the `roi_heads.fastrcnn_loss` method of the Faster R-CNN models.
* **Outcome:** By concentrating on difficult examples, the model was encouraged to learn more robust and discriminative features, which can lead to better overall accuracy and a reduction in simple errors.

### Part 4: Hard Negative Mining (HNM)

* **Goal:** Reduce the number of false positive detections.
* **Process:**
    * After an initial training phase, the model was run on the validation set to identify "hard negatives"—background regions that the model incorrectly classified as an object with high confidence.
    * These hard negative samples were then explicitly added to the training dataset with a 'background' label.
    * The model was retrained with this augmented dataset.
* **Outcome:** This process forces the model to learn to better distinguish between true objects and confusing background patterns, thereby reducing the false positive rate and improving precision.

### Part 5: Model Ensembling

* **Goal:** Combine the strengths of both the ResNet50 and MobileNetV3 models to create a more robust and accurate final prediction.
* **Process:**
    * Predictions were generated from both the final ResNet50 and MobileNetV3 models for each test image.
    * The bounding boxes and scores from both models were combined.
    * **Non-Maximum Suppression (NMS)** was applied to the combined set of predictions to merge overlapping boxes and remove redundant detections.
* **Outcome:** The ensemble model leverages the diverse feature representations learned by the two different backbones. This often leads to superior performance compared to either individual model, improving both recall and precision.

---

## ⚙️ How to Use

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/Kartik-Hiremath/dental-radiography-analysis.git](https://github.com/Kartik-Hiremath/dental-radiography-analysis.git)
    cd dental-radiography-analysis
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**
    Download the dataset from [here](https://www.kaggle.com/datasets/imtkaggleteam/dental-radiography/data) and place it in the `data` directory within your cloned repository.

5.  **Run the notebook:**
    Open the `dental_anomalies_detection_adv.ipynb` notebook and run the cells.

---

## 🛠️ Technologies Used

**Programming Language:**
* **Python:** The entire project was developed using Python, which is the most popular programming language for machine learning and data science.

**Deep Learning Framework:**
* **PyTorch:** PyTorch is an open-source machine learning framework that is widely used for developing and training deep learning models.

**Libraries:**
* **TorchVision:** A library for PyTorch that provides popular datasets, model architectures, and common image transformations for computer vision.
* **Pandas:** Used for data manipulation and analysis, specifically for loading and processing dataset annotations.
* **OpenCV (`opencv-python`):** Utilized for loading and processing images.
* **Albumentations:** Employed for applying various data augmentation techniques to training images.
* **Matplotlib:** Used for data visualization and plotting experimental results.
* **Seaborn:** Used for enhanced data visualization and plotting experimental results.
* **Scikit-learn:** Utilized for evaluating the performance of the models.

**Tools:**
* **Google Colab:** A free cloud-based service providing a Jupyter Notebook environment with free access to GPUs, used for model development and training.
* **GitHub:** A web-based platform for version control and collaboration, used for hosting and sharing the project.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions or improvements, please feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ✉️ Contact

Kartik Hiremath - kartikhiremath001@gmail.com
