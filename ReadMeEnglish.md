
## **Project Documentation: Fake News Classifier with XLM-RoBERTa**

### **Objective**
The goal of this project is to train a **Text Classification** model using **XLM-RoBERTa**, a pre-trained multilingual transformer model, to classify news as either fake (`label = 0`) or real (`label = 1`). The model is trained on a labeled dataset of news articles and evaluated on a validation set to determine its ability to predict news classes correctly.

### **Code Breakdown**

#### **1. Loading the Dataset**
First, the dataset is loaded from a CSV file. This dataset contains the news articles and their respective labels (fake or real). The CSV file is read using **pandas**, a Python library for data manipulation in tabular format (DataFrame).

```python
df = pd.read_csv(dataset_path, on_bad_lines='skip')
```

**Parameter `on_bad_lines='skip'`**: This parameter ensures that the code doesn't crash if there are malformed lines in the CSV file. It simply skips those lines.

#### **2. Splitting the Dataset**
The dataset is split into two subsets: one for training and one for validation. **`train_test_split`** from **scikit-learn** is used for this task.

- **`test_size=0.2`**: This means 20% of the data will be used for validation, while the remaining 80% will be used for training.
- **`stratify=df['label'].tolist()`**: This ensures that the proportion of classes (fake and real) is the same in both subsets (stratified splitting).

```python
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label'].tolist()
)
```

#### **3. Tokenizing the Texts**
The **XLM-RoBERTa** model requires text to be converted into tokens, i.e., numerical representations that the model can process. We use the **tokenizer** from **Hugging Face's Transformers** library to achieve this.

```python
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)
```

- **`truncation=True`**: This ensures that texts longer than 256 tokens are truncated.
- **`padding=True`**: Shorter texts are padded with a special token to ensure all texts have the same length.
- **`max_length=256`**: Limits the length of the texts to 256 tokens.

#### **4. Creating a Custom Dataset**
For the model to work with the dataset, we create a custom class `FakeNewsDataset` that inherits from `torch.utils.data.Dataset`. This class structures the data so the model can access it during training and evaluation.

```python
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx])
        return item
```

- **`__len__()`**: Returns the length of the dataset (number of samples).
- **`__getitem__()`**: Returns a dictionary containing the data for a single sample. The keys are the names of the input features (e.g., `input_ids`, `attention_mask`), and the label (`label`) is also included.

#### **5. Custom Model: `AdvancedXLMRClassifier`**
**XLM-RoBERTa** is a pre-trained **transformer** model, so it doesn't need to be trained from scratch. However, we create a custom class that extends the base **XLM-RoBERTa** model to add an additional classification layer.

##### **Freezing Layers**
We freeze the initial layers of the base model to prevent their parameters from being updated during training. This is useful because we want to leverage the pre-trained representations of the model without fine-tuning them completely.

```python
for param in self.xlm_roberta.roberta.embeddings.parameters():
    param.requires_grad = False
for param in self.xlm_roberta.roberta.encoder.layer[:5].parameters():
    param.requires_grad = False
```

- **`self.xlm_roberta.roberta.embeddings.parameters()`**: Freezes the parameters of the embedding layer (word representations).
- **`self.xlm_roberta.roberta.encoder.layer[:5].parameters()`**: Freezes the first 5 layers of the encoder.

##### **Classification Architecture**
After the **XLM-RoBERTa** layer, we add an additional multi-layer perceptron (MLP) with several **`Linear`**, **`BatchNorm1d`**, **`ReLU`**, and **`Dropout`** layers to improve model performance.

```python
self.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(self.xlm_roberta.config.hidden_size, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_labels)
)
```

- **`Dropout`**: Regularization to prevent overfitting by randomly "dropping" certain neurons during training.
- **`Linear`**: Fully connected layer that reduces the feature space.
- **`BatchNorm1d`**: Normalization of activations to stabilize training.
- **`ReLU`**: Non-linear activation function to introduce non-linearity.

##### **Forward Method**
The **`forward`** method defines how data flows through the model. First, it gets the output from **XLM-RoBERTa**, then it extracts the **[CLS] token** (which represents the entire sequence) and passes it through the additional layers of the neural network.

```python
outputs = self.xlm_roberta.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
pooled_output = outputs.last_hidden_state[:, 0, :]
logits = self.classifier(pooled_output)
```

- **`last_hidden_state[:, 0, :]`**: Selects the **[CLS] token** from the sequence (the first token) as the representation of the entire sequence.
- **`self.classifier(pooled_output)`**: Passes the **[CLS] token** through the classification layers.

#### **6. Custom Metrics Calculation**
We define the **`compute_metrics`** function to evaluate model performance using metrics like **precision**, **recall**, **F1 score**, and **accuracy**. Additionally, we experiment with different decision thresholds for predictions (0.4, 0.45, 0.5).

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.softmax(torch.tensor(logits), dim=-1)
    thresholds = [0.4, 0.45, 0.5]
    best_f1 = 0
    best_threshold = 0.45
    best_metrics = {}

    for threshold in thresholds:
        binary_predictions = (predictions[:, 1] > threshold).int()
        precision, recall, f1, _ = precision_recall_fscore_support(labels, binary_predictions, average='binary')

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                "accuracy": accuracy_score(labels, binary_predictions),
                "f1": f1,
                "precision": precision,
                "recall": recall
            }

    return best_metrics
```

#### **7. Model Training**
The **`Trainer`** manages the entire training and evaluation loop. We pass the model, data, and training parameters to it. **Early stopping** is configured to stop training if the evaluation metric does not improve after a set number of epochs.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]
)
trainer.train()
```

**`early_stopping`** is a callback function that halts training if the **F1** score does not improve after 3 consecutive epochs.

#### **8. Saving the Model**
Once the model is trained, both the model and the tokenizer are saved for future use:

```python
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')
```

#### **9. Final Evaluation**
We evaluate the model on the validation set using the **`evaluate`** method from the **Trainer**, which returns performance metrics.

```python
results = trainer.evaluate()
print("Evaluation results:", results)
```

---

### **Conclusion**
This code uses advanced techniques such as **fine-tuning** a pre-trained **XLM-RoBERTa** model, **freezing layers**, **early stopping**, and custom metric calculation to classify news articles as real or fake. The model is fine-tuned for this specific dataset, and its performance is evaluated using precision, recall, **F1 score**, and **accuracy**.

This approach

 provides an efficient and effective solution to the problem of text classification in disinformation, leveraging state-of-the-art language models.
