# Fruit Classification Using ANN

## Project Overview

This project compares the performance of ReLU and Leaky ReLU activation functions for fruit image classification using the Fruit360 dataset. The objective of this study is to evaluate the effectiveness of both activation functions in an Artificial Neural Network (ANN) model and determine the best-performing model based on classification accuracy and loss.

## Results

| Model | Accuracy | Loss |
|---------|---------:|---------:|
| ReLU | 98.34% | 0.054951 |
| Leaky ReLU | 97.92% | 0.039673 |

🏆 **Winner: ReLU**

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit
- Google Colab

## Project Structure

```text
fruit-classification-ann/
│
├── notebooks/
│   └── Fruit_Classification_ReLU_vs_LeakyReLU.ipynb
│
├── streamlit_app/
│   ├── app.py
│   ├── classes.json
│   ├── fruit_best_weights.weights.h5
│   └── requirements.txt
│
├── README.md
└── .gitignore
```

## Dataset

This project uses the Fruit360 Dataset, which contains images of various fruits captured under controlled conditions with a white background.

## Streamlit Application

The Streamlit application allows users to:

- Upload fruit images
- Predict fruit categories using the trained ANN model
- Display prediction confidence scores
- Visualize prediction probabilities
- View model comparison results

## Conclusion

The experimental results indicate that the ReLU activation function achieved the highest classification accuracy and was selected as the best-performing model. Although Leaky ReLU obtained a lower loss value, ReLU provided better overall classification performance on the Fruit360 dataset.

## Group Information

### Group 6

1. Alvita Azalia
2. Bagas Fitria Maiardi (001202507025)
3. Dasta Satria Nursyakirin (001202507029)
4. Joulle Violletta (001202507030)
5. Sutanto Hardiyanto (001202507023)
6. Winarti (001202507032)

## Course Information

BI / Image Classification Project

## License

This project is developed for academic and educational purposes.

## Institution

President University


