1. Introduction (samed)
   
  Soziale Medien spielen heutzutage eine zentrale Rolle im Leben vieler Menschen. Dabei ist die Verbreitung von Hate speech zu einem großen Problem geworden. Laut einer Umfrage in Deutschland im Jahr 2022 zur Wahrnehmung von Hate Speech im Internet haben 78% der Befragten schon mal Hassrede im Internet gesehen. Bei 14-24 Jährigen sind es sogar 92% (Statista, 2024). Soziale Medien ermöglichen zwar freie Meinungsäußerungen, jedoch wird dies oft missbraucht, um Hass zu verbreiten. Diese negativen Inhalte können sowohl psychologische als auch soziale Schäden verursachen. Es ist praktisch unmöglich Hassreden manuell zu erkennen und zu beseitigen bzw. zu verhindern. Hierbei ist es notwendig, automatisierte Systeme zur Erkennung von Hate speech einzusetzen. 
  Unsere Forschungsfrage lautet daher: Wie können maschinelle Lernmethoden effektiv eingesetzt werden, um Hate Speech in sozialen Medien zu erkennen und zu klassifizieren? 

  Das Dokument ist wie folgt strukturiert:
  1. Introduction
  2. Related Work
  3. Methodology
  4. Results
  5. Discussion
  6. Conclusion
  7. bibliography
   
2. Related Work (samed)    
   
   In mehreren Studien wurde der Einsatz von Machine Learning und Neuronalen Netzen zur Erkennung von Hassreden untersucht. Putra et al erreichen bei der Klassifizierung von Hate Speech in Hassrede oder Nicht-Hassrede mittels Convolutional Neuronal Network eine hohe Genauigkeit (accuracy) von ca. 95.89% (Putra et al., 2022). Hüsünbeyi et al heben hervor, dass das Bidirectional Encoder Representations (BERT) für kontextlastige Texte eine leistungsstarke Lösung darstellt. Sie erreichen zusammen mit der Hierarchical Attention Network (HAN) ebenfalls eine hohe Genauigkeit (Hüsünbeyi et al., 2022). Subramanian et al stellen die Herausforderungen bei der Erkennung von Hassreden und Stimmungsanalyse (sentiment analysis) dar. Zum einen sind die Nachrichten in sozialen Medien nicht immer strukturiert und teils schlecht geschrieben. Dies erschwert das Erkennen von Mustern in den Texten. Zudem können Hassreden subjektiv sein und vom Kontext abhängen, was die Erkennung von Hassreden erschwert (Subramanian et al., 2023).

3. Methodology
   
   3.1. General Methodology (luis)
   
      How did you proceed to achieve your project goals?
      Describe which steps you have undertaken
      Aim: Others should understand your research process

   3.2. Data Understanding and Preparation (Marie) 
   
      Introduce the dataset to the reader
      Describe structure and size of your dataset

      Structure and Size of the Dataset
In our use case, we aim to develop a classification algorithm capable of categorizing text-based data into various emotional categories. To achieve this, we require datasets that contain specific labels to facilitate the training and evaluation of our model. Our selected dataset consists of 120,000 rows and three columns: ['tweet_id', 'sentiment', 'content']. The data types are int64 for 'tweet_id' and object for 'sentiment' and 'content'. The 'content' column contains the tweets, while the 'sentiment' column holds the labels, which include categories such as empty, sadness, enthusiasm, neutral, worry, surprise, love, fun, hate, happiness, boredom, relief, and anger.


      Describe how you prepare the dataset for your project
As outlined, we focus on classifying five primary sentiment categories. To ensure a comprehensive data foundation, we selected the five most frequently represented labels and included the "hate" category due to its significant representation of emotion in tweets. Our analysis revealed an uneven distribution of data in the dataset. To ensure a more balanced distribution for model training, we limited the number of data points per sentiment to the level of the "sadness" category. This approach resulted in the following distribution:

Although the "love" and "hate" categories remain underrepresented, we consciously opted for this trade-off to maintain a sufficient data volume for model training. This strategy helps optimize the balance between data diversity and volume, enhancing the model's robustness.

   
   3.3. Modeling and Evaluation (Luis)
   
      Describe the model architecture(s) you selected
      Describe how you train your models
      Describe how you evaluate your models/ which metrics you use

5. Results (marie)
   
   Describe what artifacts you have build
   Describe the libraries and tools you use
   Describe the concept of your app
   Describe the results you achieve by applying your trained models on unseen data
   Descriptive Language (no judgement, no discussion in this section -> just show what you built)
   
6. Discussion (maike)
   
   Now its time to discuss your results/ artifacts/ app
   Show the limitations : e.g. missing data, limited training ressources/ GPU availability in Colab, limitaitons of the app
   Discuss your work from an ethics perspective:
   Dangers of the application of your work (for example discrimination through ML models)
   Transparency
   Effects on Climate Change
   Possible sources https://algorithmwatch.org/en/ Have a look at the "Automating Society Report"; https://ainowinstitute.org/ Have a look at this website and their
   publications
   Further Research: What could be next steps for other researchers (specific research questions)
   
7. Conclusion (Maike)
    
   Short summary of your findings and outlook

8. bibliography


   Hatespeech: Wahrnehmung nach Alter 2022. (2024). Statista. Abgerufen 2. Juni 2024, von https://de.statista.com/statistik/daten/studie/1365862/umfrage/umfrage-in-deutschland-zur-wahrnehmung-von-hate-speech-im-internet-nach-alter/

   Hüsünbeyi, Z. M., Akar, D., & Özgür, A. (2022). Identifying Hate Speech Using Neural Networks and Discourse Analysis Techniques. LATERAISSE. https://www.semanticscholar.org/paper/Identifying-Hate-Speech-Using-Neural-Networks-and-H%C3%BCs%C3%BCnbeyi-Akar/4bb1247c0de893a014a3a983bcb5c8cdf2717065

   Putra, B. P., Irawan, B., Setianingsih, C., Rahmadani, A., Imanda, F., & Fawwas, I. Z. (2022). Hate Speech Detection using Convolutional Neural Network Algorithm Based on Image. 2021 International Seminar on Machine Learning, Optimization, and Data Science (ISMODE), 207–212. https://doi.org/10.1109/ISMODE53584.2022.9742810

   Subramanian, M., Easwaramoorthy Sathiskumar, V., Deepalakshmi, G., Cho, J., & Manikandan, G. (2023). A survey on hate speech detection and sentiment analysis using machine learning and deep learning models. Alexandria Engineering Journal, 80, 110–121. https://doi.org/10.1016/j.aej.2023.08.038




