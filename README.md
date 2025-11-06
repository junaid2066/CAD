# Conv-LSTM for Real-Time Spatio-Temporal Analysis of Crowd Behavior in Public Spaces
This project focuses on **real-time violence detection** in video datasets using deep learning-based visual models.  
Two benchmark datasets were used for experimentation:

1. **Real-Life Violence Situation Dataset (RLVS)** – for initial evaluation.  
2. **Hockey Fight Dataset (HFD)** – for final model validation.

The models were trained to differentiate between violent and non-violent actions using convolutional neural networks (CNNs) and transfer learning strategies.

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

## 📘 Overview


---

## 🧠 Methodology

The proposed pipeline consists of the following key steps:

- **Frame Extraction:** Frames were sampled at uniform intervals from input videos.
- **Feature Extraction:** Deep visual features were extracted using pre-trained CNN backbones (e.g., ResNet, VGG16).
- **Classification:** Fully connected layers were trained to predict violence probability per frame and video.
- **Result Fusion:** Temporal averaging was used to combine frame-level predictions into final decisions.

---

## 📊 Results

### 🧩 Real-Life Violence Dataset (Initial Results)
- **Accuracy:** 92.4%  
- **Precision:** 91.7%  
- **Recall:** 90.5%  
- **F1-Score:** 91.1%  

### 🏒 Hockey Fight Dataset (Final Results)
- **Accuracy:** 95.8%  
- **Precision:** 94.9%  
- **Recall:** 95.5%  
- **F1-Score:** 95.2%  

The results demonstrate strong generalization across datasets, confirming the model’s robustness and efficiency in recognizing violent actions in real-world videos.

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

To run the notebooks:

```bash
jupyter notebook "RealLifeViolenceDS(Initial Results).ipynb"
jupyter notebook "Hockey Fight Dataset (Final Results).ipynb"
```

You can also adapt the code to your own video datasets by modifying the frame extraction and preprocessing cells.

---

## 📁 Repository Structure

```
Violence-Detection-Using-Deep-Learning/
│
├── RealLifeViolenceDS(Initial Results).ipynb
├── Hockey Fight Dataset (Final Results).ipynb
├── requirements.txt
└── README.md
```

---

## 📖 Citation

If you use this repository or refer to this work, please cite as:

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

## 🧾 Repository Citation

If you use this repository in your research or projects, please cite it as:

> Muhammad Junaid Asif, *Violence Detection Using Deep Learning*, GitHub Repository, 2025.  
> Available at: [https://github.com/mjunaidasif/Violence-Detection-Using-Deep-Learning](https://github.com/mjunaidasif/Violence-Detection-Using-Deep-Learning)

---

## ✍️ Author

👨‍💻 **Developed by Muhammad Junaid Asif**  
🔗 GitHub: [@mjunaidasif](https://github.com/mjunaidasif)  
📧 Email: mjunaidasif@gmail.com

---

