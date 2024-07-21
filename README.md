# Prüfungsaufgabe 2

**Anleitung zum ausführen des Notebooks:**

**Update:**

Laden & aufrufen des Notebooks in MyBinder dauert bis zu 20 Minuten. 
Alternative gesucht & auf Google Colab gestoßen.
Ausführen in Google Colab ebenfalls möglich & performatnter:
1. Klicken Sie auf den Google Colab Batch ↓
2. Warten Sie bis sich das Notebook geöffnet hat (maximal 10sek)
3. Das Notebook wird direkt aufgerufen und kann durch verwenden des Play-Button schrittweise ausgeführt werden

<a target="_blank" href="https://colab.research.google.com/github/MichaelFranLu/Pruefungsaufgabe_2_MNIST_Dataset/blob/main/Pru%CC%88fungsaufgabe_2_MNIST-Dataset.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<a target="_blank" href="https://colab.research.google.com/github/MichaelFranLu/Pruefungsaufgabe_2_MNIST_Dataset?tab=readme-ov-file">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Einleitung
In dieser Lektion erstellen wir ein **Multi Layer Perceptron Modell** mit dem Ziel, handgeschriebene Zahlen zu klassifizieren. Dies ist ein sehr verbreitetes Einsteigerproblem für **Tensorflow**. Der Code implementiert eine Anwendung zur Verarbeitung von MNIST-Daten und zur Erstellung dieses maschinellen Lernmodells. Es ist wichtig zu beachten, dass eine einzige Lektion niemals ausreichen wird, um **Deep Learning** und/oder **Tensorflow** in seiner Komplexität abzudecken.

## Datenverarbeitung
1. **Bibliotheken importieren:** Der Code beginnt mit dem Importieren der erforderlichen Python-Bibliotheken wie TensorFlow, NumPy, Logging, Time, Matplotlib und Unittest.
2. **MNIST-Daten verwenden:** Der Code verwendet den MNIST-Datensatz, der aus 28x28 Pixel großen schwarz-weiß Bildern von handgeschriebenen Zahlen besteht. Das Ziel ist es, die auf den Bildern dargestellten Zahlen korrekt vorherzusagen.
3. **Datenformat:** Die Originaldaten, die eine 2D-Matrix waren, sind im Vektorformat gespeichert. Es gibt einige Untersuchungen zum Datenformat und zur Darstellung eines Beispieldatensatzes.
4. **Verarbeitung des MNIST-Datensatzes:** Der Code normalisiert und formt die Trainings- und Testbilder um, sodass sie als eindimensionale Arrays dargestellt werden. Anschließend teilt er die Daten in Trainings- und Testsets auf, wobei 80% der Daten für das Training und 20% für das Testen verwendet werden. Zuletzt überprüft er, ob die Form der Daten korrekt ist.

## Integration der Funktion my_logger, my_Timer & der Klasse TheAlgorithm
In den folgenden Abschnitten werden wir uns mit Unit Testing und Logging befassen. Diese Konzepte sind entscheidend für die Erstellung robuster, zuverlässiger und wartbarer Codebasen und sind in der Softwareentwicklung weit verbreitet. In der Datenwissenschaft können sie jedoch oft vernachlässigt werden. Die nächsten drei Code-Abschnitte basieren auf den Prinzipien und Praktiken, die in einem Artikel auf ‘Unit Testing and Logging for Data Science’ vorgestellt wurden, und wurden für unseren spezifischen Anwendungsfall angepasst. 
→ https://towardsdatascience.com/unit-testing-and-logging-for-data-science-d7fb8fd5d217

1. **Dekoratoren:** Zwei Dekoratoren, my_logger und my_timer, sind definiert. my_logger protokolliert die Argumente, mit denen eine Funktion aufgerufen wird, und my_timer misst die Ausführungszeit einer Funktion.
2. **Klassendefinition:** Die Klasse TheAlgorithm ist definiert, die Methoden zum Trainieren (fit) und Testen (predict) eines maschinellen Lernmodells enthält.
3. **Initialisierung:** Im Konstruktor der Klasse werden die Trainings- und Testdaten gespeichert.
4. **Training:** In der fit-Methode wird ein LogisticRegression-Modell erstellt und mit den Trainingsdaten trainiert. Die Daten werden vor dem Training normalisiert. Die Genauigkeit des Modells auf den Trainingsdaten wird berechnet und zurückgegeben.
5. **Vorhersage:** In der predict-Methode wird das trainierte Modell verwendet, um Vorhersagen auf den Testdaten zu machen. Die Genauigkeit des Modells auf den Testdaten wird berechnet und zurückgegeben.

## Multi-Layer Perceptron (MLP)
1. **Parameterdefinition:** Es werden verschiedene Parameter für das Modell und das Training definiert, wie z.B. die Lernrate, die Anzahl der Trainingsepochen, die Batch-Größe und die Netzwerkparameter.
2. **TensorFlow Graph Input:** Es werden Input-Tensoren für die Datenpunkte (x) und die Klassenlabels (y) erstellt.
3. **Multi-Layer Perceptron Funktion:** Diese Funktion erstellt das MLP-Modell. Es hat zwei versteckte Schichten mit ReLU-Aktivierung und eine Ausgabeschicht mit linearer Aktivierung.
4. **Gewichts- und Bias-Initialisierung:** Die Gewichte und Biases für die Schichten des MLP werden initialisiert.
5. **Modellerstellung:** Das MLP-Modell wird erstellt, indem die multilayer_perceptron Funktion aufgerufen wird.
6. **Kosten- und Optimierungsfunktion:** Die Kostenfunktion ist die durchschnittliche Softmax-Kreuzentropie zwischen den Labels und den Vorhersagen des Modells. Der Optimierer ist der Adam-Optimierer mit der zuvor definierten Lernrate.

## Modelltraining
1. **Visualisierung eines Beispieldatensatzes:** Eine Funktion wird definiert, um ein Beispiel aus dem Datensatz zu visualisieren.
2. **Logger und Timer Dekoratoren:** Zwei Dekoratoren werden definiert, um Informationen über die Ausführung von Funktionen zu protokollieren und die Ausführungszeit zu messen.
3. **Training des Modells:** Das MLP-Modell wird für eine bestimmte Anzahl von Epochen trainiert. In jeder Epoche wird das Modell auf jedem Batch der Trainingsdaten trainiert und die durchschnittlichen Kosten (Verlust) werden berechnet.

Im gegebenen Code wird das Training des Modells auf nur **0,5% der gesamten Trainingsdaten** durchgeführt. Dies wurde hauptsächlich aus **Performance-Gründen (Performance des Trainingsprozesses)** entschieden, um die Ausführungszeit des Codes zu reduzieren. **Es ist jedoch wichtig zu beachten, dass diese Entscheidung einen Kompromiss darstellt: Während wir eine schnellere Ausführung erreichen, könnte die Qualität des trainierten Modells darunter leiden, da es auf einer stark reduzierten Menge an Daten trainiert wird.**

## Neuronales Netzwerk
In diesem Codeabschnitt wird ein neuronales Netzwerkmodell mit TensorFlow’s Keras API erstellt, kompiliert, trainiert und evaluiert. Hier sind die wichtigsten Schritte:

1. **Modellerstellung:** Ein sequentielles Modell wird erstellt, das aus mehreren Schichten besteht, darunter zwei dichte (Dense) Schichten mit ReLU-Aktivierung und Batch-Normalisierung, und einer Ausgabeschicht ohne Aktivierung.
2. **Modellkompilierung:** Das Modell wird mit dem Adam-Optimierer, der Sparse Categorical Crossentropy als Verlustfunktion und der Sparse Categorical Accuracy als Metrik kompiliert.
3. **Callbacks:** Zwei Callbacks werden definiert: EarlyStopping, das das Training stoppt, wenn die Validierungsverluste nicht mehr verbessert werden, und ModelCheckpoint, das das beste Modell während des Trainings speichert.
4. **Modelltraining:** Das Modell wird auf den Trainingsdaten trainiert, die zuvor in ein flaches Format umgewandelt wurden. Ein Teil der Trainingsdaten wird als Validierungsdatensatz verwendet.
5. **Modellevaluierung:** Das Modell wird auf den Testdaten evaluiert, die ebenfalls in ein flaches Format umgewandelt wurden. Der Testverlust und die Testgenauigkeit werden berechnet und ausgegeben.

## Modellbewertung - Prüfungsaufgabe 1&2
Die nächsten drei Code-Abschnitte basieren auf den Prinzipien und Praktiken, die in einem Artikel auf ‘Unit Testing and Logging for Data Science’ vorgestellt wurden, und wurden für unseren spezifischen Anwendungsfall angepasst. 
→ https://towardsdatascience.com/unit-testing-and-logging-for-data-science-d7fb8fd5d217

**1.Testfall:** Wählen Sie geeignete Indikatoren, die Ihnen anzeigen, dass die Vorhersagefunktion
predict() des Modells korrekt funktioniert. Im Artikel nutzt der Autor die Indikatoren
Accuracy und Confusion Matrix. Schreiben Sie für predict() einen Python Unittest Testfall auf
ausgewählten Testdaten. Die Testdaten legen Sie in einem Testdatenfile ab.

+ **test_predict:** Dieser Test überprüft, ob die predict-Methode von TheAlgorithm korrekt funktioniert. Er stellt sicher, dass die predict-Methode die erwartete Testgenauigkeit zurückgibt und dass die resultierende Test Confusion Matrix der erwarteten entspricht. Auf Basis des Code-Snippets & und Auswahl der gleichen Indikatoren (Test Accuracy & Test Confusion).

**2. Testfall:** Überprüfen Sie, dass das System innerhalb normaler Parameter läuft, indem Sie die
Laufzeit der Trainingsfunktion fit() in einem Python Unittest testen. Loggen Sie dazu eine
repräsentative Laufzeit mit dem obigen Wrapper. In dem Testfall überprüfen Sie, dass die
Laufzeit der Trainingsfunktion während der Testfallausführung einen Grenzwert, z.B. 120%
der repräsentativen Laufzeit, nicht überschreitet.

+ **test_predict:** Dieser Test überprüft, ob die predict-Methode von TheAlgorithm korrekt funktioniert. Er stellt sicher, dass die predict-Methode die erwartete Testgenauigkeit zurückgibt und dass die resultierende Test Confusion Matrix der erwarteten entspricht. Auf Basis des Code-Snippets.

+ **test_duration:** Dieser Test überprüft, ob die fit-Methode von TheAlgorithm innerhalb einer akzeptablen Zeit ausgeführt wird. Die akzeptable Zeit wird als 120% der repräsentativen Laufzeit definiert. (Neu definierte Funktion im UNI-Test).

## Ergebnisse - Prüfungsaufgabe 1&2

+ **Test_fit:** Der Test hat bestätigt, dass die Genauigkeit des Trainingsprozesses mit der erwarteten Trainingsgenauigkeit übereinstimmt. Die Confusion Matrix des Trainingsprozesses stimmt ebenfalls mit der erwarteten Confusion Matrix überein. Die Ausführungszeit für diesen Test betrug 14.9 Sekunden.
+ **Test_predict:** Der Test hat bestätigt, dass die Genauigkeit der Vorhersage mit der erwarteten Testgenauigkeit übereinstimmt. Die Confusion Matrix der Vorhersage stimmt ebenfalls mit der erwarteten Confusion Matrix überein. Die Ausführungszeit für diesen Test betrug 13.5 Sekunden.
+ **Test_duration:** Dieser Test überprüft, ob die Ausführungszeit des Trainingsprozesses innerhalb eines akzeptablen Bereichs liegt, der als 120% der repräsentativen Laufzeit definiert wurde. Der Test hat bestätigt, dass die tatsächliche Ausführungszeit des Trainingsprozesses innerhalb dieses Bereichs liegt. Die Ausführungszeit für diesen Test betrug 13.5 Sekunden.

→ Alle drei Tests erfolgreich ausgeführt und haben insgesamt 42.7 Sekunden gedauert.
→ Die Gesamtgenauigkeit des Modells beträgt 73%.

