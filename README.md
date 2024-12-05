
## **Documentación del Proyecto: Clasificador de Noticias Falsas con XLM-RoBERTa**

### **Objetivo**
Este proyecto tiene como objetivo entrenar un modelo de **Clasificación de Texto** utilizando **XLM-RoBERTa**, un modelo preentrenado de **transformers** multilingües, para clasificar noticias como falsas (`label = 0`) o verdaderas (`label = 1`). El modelo se entrena utilizando un conjunto de datos etiquetado de noticias y se evalúa en un conjunto de validación para determinar su capacidad para predecir correctamente las clases de las noticias.

### **Desglose del Código**

#### **1. Cargar el Conjunto de Datos**
En primer lugar, se carga el conjunto de datos desde un archivo CSV. Este dataset contiene las noticias y sus respectivas etiquetas (falsas o verdaderas). El archivo CSV se lee usando **pandas**, una librería de Python para manipular datos en formato de tabla (DataFrame).

```python
df = pd.read_csv(dataset_path, on_bad_lines='skip')
```

**Parámetro `on_bad_lines='skip'`**: Este parámetro se utiliza para evitar que el código se caiga si hay líneas mal formadas en el archivo CSV. Simplemente las omite.

#### **2. División del Conjunto de Datos**
El conjunto de datos se divide en dos subconjuntos: uno para entrenamiento y otro para validación. **`train_test_split`** de **scikit-learn** se utiliza para esta tarea.

- **`test_size=0.2`**: Indica que el 20% de los datos se usarán para validación, mientras que el 80% restante se destinará al entrenamiento.
- **`stratify=df['label'].tolist()`**: Asegura que las proporciones de las clases (falsas y verdaderas) sean iguales en ambos subconjuntos (estratificación).

```python
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label'].tolist()
)
```

#### **3. Tokenización de los Textos**
El modelo **XLM-RoBERTa** requiere que los textos se conviertan en tokens, es decir, en una representación numérica que el modelo pueda procesar. Utilizamos el tokenizador de **Hugging Face's Transformers**.

```python
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)
```

- **`truncation=True`**: Si un texto es más largo que el máximo de 256 tokens, se trunca.
- **`padding=True`**: Si un texto es más corto, se rellena con un token especial para asegurarse de que todos los textos tengan la misma longitud.
- **`max_length=256`**: Limita la longitud de los textos a 256 tokens.

#### **4. Creación de un Dataset Personalizado**
Para que el modelo pueda trabajar con el conjunto de datos, se crea una clase personalizada `FakeNewsDataset` que hereda de `torch.utils.data.Dataset`. Esta clase estructura los datos para que el modelo pueda acceder a ellos durante el entrenamiento y la evaluación.

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

- **`__len__()`**: Devuelve la longitud del conjunto de datos (número de muestras).
- **`__getitem__()`**: Devuelve un diccionario con los datos de una muestra. Las claves son los nombres de las columnas de los tensores (`input_ids`, `attention_mask`), y se agrega también la etiqueta (`label`).

#### **5. Modelo Personalizado: `AdvancedXLMRClassifier`**
El modelo **XLM-RoBERTa** es un modelo preentrenado de **transformers**, por lo que no es necesario entrenarlo desde cero. Sin embargo, creamos una clase personalizada que extiende el modelo base de **XLM-RoBERTa** para añadir una capa de clasificación adicional.

##### **Congelación de Capas**
Se congelan las primeras capas del modelo base para evitar que sus parámetros se actualicen durante el entrenamiento. Esto es útil para aprovechar las representaciones preentrenadas del modelo sin tener que ajustarlas completamente.

```python
for param in self.xlm_roberta.roberta.embeddings.parameters():
    param.requires_grad = False
for param in self.xlm_roberta.roberta.encoder.layer[:5].parameters():
    param.requires_grad = False
```

- **`self.xlm_roberta.roberta.embeddings.parameters()`**: Congela los parámetros de la capa de embeddings (representaciones de palabras).
- **`self.xlm_roberta.roberta.encoder.layer[:5].parameters()`**: Congela las primeras 5 capas del encoder.

##### **Arquitectura de Clasificación**
Después de la capa de **XLM-RoBERTa**, añadimos una red neuronal adicional (MLP) con varias capas **`Linear`**, **`BatchNorm1d`**, **`ReLU`** y **`Dropout`** para mejorar el rendimiento del modelo.

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

- **`Dropout`**: Regularización para prevenir el sobreajuste, que apaga aleatoriamente ciertas neuronas durante el entrenamiento.
- **`Linear`**: Capa totalmente conectada que reduce la dimensión del espacio de características.
- **`BatchNorm1d`**: Normalización de las activaciones para estabilizar el entrenamiento.
- **`ReLU`**: Función de activación no lineal para introducir no linealidad.

##### **Método Forward**
El método **`forward`** es el que define cómo pasan los datos a través del modelo. Primero, obtiene las salidas de **XLM-RoBERTa**, luego toma el **[CLS] token** (el token que representa la secuencia completa) y lo pasa a través de las capas adicionales de la red neuronal.

```python
outputs = self.xlm_roberta.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
pooled_output = outputs.last_hidden_state[:, 0, :]
logits = self.classifier(pooled_output)
```

- **`last_hidden_state[:, 0, :]`**: Selecciona el **[CLS] token** de la secuencia (el primer token) como representación de toda la secuencia.
- **`self.classifier(pooled_output)`**: Pasa el token **[CLS]** por la red de clasificación.

#### **6. Cálculo de Métricas Personalizadas**
Se define la función **`compute_metrics`** para evaluar el rendimiento del modelo usando métricas como **precisión**, **recall**, **F1 score** y **accuracy**. Además, se experimenta con diferentes umbrales de decisión para las predicciones (0.4, 0.45, 0.5).

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

#### **7. Entrenamiento del Modelo**
El **`Trainer`** es el encargado de gestionar el ciclo completo de entrenamiento y evaluación. Se le pasa el modelo, los datos, y los parámetros de entrenamiento. También se configura el **early stopping** para detener el entrenamiento si la métrica de evaluación no mejora después de un número definido de épocas.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute

_metrics,
    callbacks=[early_stopping]
)
trainer.train()
```

**`early_stopping`** es una función de callback que detiene el entrenamiento si la métrica **F1** no mejora después de 3 épocas consecutivas.

#### **8. Guardado del Modelo**
Una vez entrenado el modelo, se guarda tanto el modelo como el tokenizador para su reutilización posterior:

```python
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')
```

#### **9. Evaluación Final**
Se evalúa el modelo en el conjunto de validación usando la función **`evaluate`** del **Trainer**, que devuelve las métricas de rendimiento.

```python
results = trainer.evaluate()
print("Resultados de evaluación:", results)
```

---

### **Conclusión**
Este código utiliza técnicas avanzadas como **fine-tuning** de un modelo preentrenado de **XLM-RoBERTa**, congelación de capas, **early stopping**, y cálculo de métricas personalizadas para clasificar noticias como verdaderas o falsas. El modelo es afinado para este conjunto de datos específico, y su rendimiento se evalúa con precisión, recall, **F1 score** y **accuracy**.

Este enfoque proporciona una solución eficiente y efectiva para el problema de clasificación de texto en problemas de desinformación, utilizando modelos de lenguaje de última generación.
