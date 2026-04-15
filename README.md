# Conv-LSTM for Real-Time Spatio-Temporal Analysis of Crowd Behavior in Public Spaces
This project focuses on **real-time violence detection** in video datasets using deep learning-based visual models.  
Two benchmark datasets were used for experimentation:

1. **Real-Life Violence Situation Dataset (RLVS)** – for initial evaluation.  
2. **Hockey Fight Dataset (HFD)** – for final model validation.

The models were trained to differentiate between violent and non-violent actions using convolutional neural networks (CNNs) and transfer learning strategies.

## Proposed Methodology
<img width="521" height="384" alt="Fig 02  Proposed Model for Crowd Anomaly Detection" src="https://github.com/user-attachments/assets/08f95a8d-28ac-4211-b683-a9bf697ae51b" />


-----
**Research Publication**  
**"Conv-LSTM for Real-Time Spatio-Temporal Analysis of Crowd Behavior in Public Spaces"**  
Published in the *Proceedings of the IEEE 4th International Conference on Communication, Computing and Digital Systems (C-CODE)*, 2025.  
DOI: [10.1109/11204064](https://ieeexplore.ieee.org/document/11204064)

**Citation:**
```bash
@INPROCEEDINGS{11204064,
  author={Asif, Muhammad Junaid and Saqib, Shazia and Ahmad, Rana Fayyaz and Asad, Mujtaba and Hussain Rizvi, Syed Tahir},
  booktitle={2025 4th International Conference on Communication, Computing and Digital Systems (C-CODE)}, 
  title={Conv-LSTM for Real-Time Spatio-Temporal Analysis of Crowd Behavior in Public Spaces}, 
  year={2025},
  volume={},
  number={},
  pages={1-9},
  keywords={Deep learning;Image analysis;Surveillance;Roads;Feature extraction;Public security;Real-time systems;Reliability;Long short term memory;Sports;Crowd scene analysis;Behavior analysis;Anomaly detection;Conv-LSTM;VGG19;LSTM;Wide Dense Residual block},
  doi={10.1109/C-CODE67372.2025.11204064}}
```

---

## 🧩 Modular Code Structure

The repository is organized into clear, reusable Python modules for easier maintenance and extension.

```
Violence-Detection-Using-Deep-Learning/
│
├── RealLifeViolenceDS(Initial Results).ipynb
├── Hockey Fight Dataset (Final Results).ipynb
│
├── modules/
│   ├── preprocessing.py        # Frame extraction, resizing, normalization
│   ├── model.py                # Model architecture (CNN / ResNet / MobileNetV2)
│   ├── train.py                # Training script (augmentation, callbacks, optimizer)
│   ├── test.py                 # Evaluation metrics and testing
│   ├── plot_results.py         # Visualization: accuracy/loss curves & confusion matrix
│
├── main.py                     # Unified entrypoint combining preprocessing → training → testing
├── requirements.txt
└── README.md
```
---
---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/mjunaidasif/Violence-Detection-Using-Deep-Learning.git
cd Violence-Detection-Using-Deep-Learning
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1️⃣ Preprocess Video Dataset

```bash
python modules/preprocessing.py --data_path /path/to/dataset
```

### 2️⃣ Train the Model

```bash
python modules/train.py --dataset /path/to/processed/data --model resnet
```

### 3️⃣ Test and Evaluate

```bash
python modules/test.py --weights /path/to/saved_model.h5
```

### 4️⃣ Plot Results

```bash
python modules/plot_results.py --history /path/to/training_history.json
```

### 5️⃣ Run Complete Pipeline

```bash
python main.py
```

---

## 📊 Experimental Results

### 🧩 Real-Life Violence Dataset (Initial Results)
- **Accuracy:** 93.8%  
- **Precision:** 0.93 
- **Recall:** 0.92
- **F1-Score:** 0.90

### 🏒 Hockey Fight Dataset (Final Results)
- **Accuracy:** 91.0%  
- **Precision:** 0.91 
- **Recall:** 0.82 
- **F1-Score:** 0.86 

These results confirm the model’s robustness across diverse environments and datasets.

---

## 🧠 Key Features

- Modular and readable structure for reproducibility  
- Transfer learning support (ResNet50, MobileNetV2, custom CNN)  
- Evaluation on multiple datasets  
- Visualization utilities for training and performance metrics  
- Ready for integration with real-time systems

---

## 🧾 Citation

If you use this repository or refer to this work, please cite the paper as:

```
@INPROCEEDINGS{11204064,
  author={Asif, Muhammad Junaid and others},
  title={Real-Time Violence Detection Using Deep Learning},
  booktitle={Proceedings of the IEEE International Conference on Artificial Intelligence and Machine Vision (AIMV)},
  year={2025},
  doi={10.1109/11204064}
}
```

---

## 📖 Repository Citation

If you use this repository in your research or projects, please cite it as:

> Muhammad Junaid Asif, *Violence Detection Using Deep Learning*, GitHub Repository, 2025.  
> Available at: [https://github.com/mjunaidasif/Violence-Detection-Using-Deep-Learning](https://github.com/mjunaidasif/Violence-Detection-Using-Deep-Learning)

---

## 🔒 Access to Code
If you want access to the code, click below:
[📩 Request Code Access](mailto:muhammadjunaid94ee@gmail.com?subject=Request%20for%20Code%20Access&body=Hello,%20I%20am%20interested%20in%20your%20project%20Titled:CAD:%20A%20Crowd%20Anomaly%20Detection.%20Please%20share%20access.)

-----

## ✍️ Author

👨‍💻 **Developed by Muhammad Junaid Asif**  
🔗 GitHub: [@mjunaidasif](https://github.com/mjunaidasif)  
📧 Email: mjunaid94ee@outlook.com

---
