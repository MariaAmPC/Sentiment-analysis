1. Introduction (samed)
   
  Today, social media plays a central role in the exchange of opinions and information. Sentiment analysis is becoming increasingly important. Companies, political actors and researchers use this technology to capture and analyse public sentiment and opinions in real time. 
  Against this background, the research question is: How can machine learning methods be used effectively to classify sentiment using sentiment analysis in social media?


  The document is structured as follows:
  1. Introduction
  2. Related Work
  3. Methodology
  4. Results
  5. Discussion
  6. Conclusion
  7. bibliography
   
2. Related Work (samed)    
   
   The use of machine learning and neural networks for sentiment analysis has been investigated in several studies. Neethu and Rajasree present the different classification techniques used for sentiment analysis. These are Nave Bayes Classifier, SVM Classifier, Maximum Entropy Classifier and Ensemble Classifier, all of which perform similarly. Naive Bayes has better accuracy, but the other classifiers have better accuracy and recognition (Neethu & Rajasree, 2013). 
   Duyu Tang et al point out two main directions of sentiment classification: lexicon-based approach and corpus-based approach. Lexicon-based approaches use dictionaries with predefined sentiment values and integrate linguistic features such as negation and intensification to calculate the sentiment polarity of texts. Corpus-based methods, on the other hand, use machine learning with annotated data sets and various features, with the best results often being achieved by Support Vector Machines (SVM) with bag-of-words features (Duyu Tang et al., 2015).
   Wankhade et al presented the challenges of sentiment analysis. Dealing with ambiguous situations and irony is a central difficulty in sentiment analysis. It is a challenge for a machine to understand cultural allusions. In addition, performance in sentiment analysis can be influenced by grammatical or spelling errors in the text (Wankhade et al., 2022).

3. Methodology
   
   3.1. General Methodology (luis)
   
      How did you proceed to achieve your project goals?
      Describe which steps you have undertaken
      Aim: Others should understand your research process

   3.2. Data Understanding and Preparation (Marie) 
   
In our use case, we aim to develop a classification algorithm capable of categorizing text-based data into various emotional categories. To achieve this, we require datasets that contain specific labels to facilitate the training and evaluation of our model. Our selected dataset consists of 120,000 rows and three columns: ['tweet_id', 'sentiment', 'content']. The data types are int64 for 'tweet_id' and object for 'sentiment' and 'content'. The 'content' column contains the tweets, while the 'sentiment' column holds the labels, which include categories such as empty, sadness, enthusiasm, neutral, worry, surprise, love, fun, hate, happiness, boredom, relief, and anger.

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

   In our project we have developed an application capable of categorizing tweets to predict if people will react on the text with sadness, neutral, worried or if they hate it and so on. Our model was trained with on a dataset consisting of 120.000 tweets. 
Despite the comprehensiveness of the dataset, it´s important to acknowledge that some relevant tweets may have been missed and initially, we faced limitations in terms of training resources and GPU availability in Colab.
One of the challenges we encountered was the potential for inadvertent discrimination introduced by the algorithm. Biased language within the dataset could lead to inaccurate predictions. Therefore we took great care in selecting a dataset that was diverse and comprehensive, and had various examples of tweets to mitigate this issue.

   
   Now its time to discuss your results/ artifacts/ app
   Show the limitations : e.g. missing data, limited training ressources/ GPU availability in Colab, limitaitons of the app
   Discuss your work from an ethics perspective:
   Dangers of the application of your work (for example discrimination through ML models)
   Transparency
   Effects on Climate Change
   Possible sources https://algorithmwatch.org/en/ Have a look at the "Automating Society Report"; https://ainowinstitute.org/ Have a look at this website and their
   publications
   Further Research: What could be next steps for other researchers (specific research questions)
   
8. Conclusion (Maike)

   In conclusion, our project aimed to develop a high-quality application for sentiment analysis of tweets and this will be achieved through accurate data processing. 
Moving forward, our application holds significant potential to assist users in understanding how their tweets will be received by others, thanks to high-quality predictions generated by our model. By prioritizing accuracy, our project contributes to enhancing the overall quality and utility of sentiment analysis tools in the realm of social media.

    
   Short summary of your findings and outlook

10. bibliography


   Duyu Tang, Bing Quin, & Ting Liu. (o. J.). Deep learning for sentiment analysis: Successful approaches and future challenges—Tang—2015—WIREs Data Mining and Knowledge Discovery—Wiley Online Library. Abgerufen 2. Juni 2024, von https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1171

   Neethu, M. S., & Rajasree, R. (2013). Sentiment analysis in twitter using machine learning techniques. 2013 Fourth International Conference on Computing, Communications and Networking Technologies (ICCCNT), 1–5. https://doi.org/10.1109/ICCCNT.2013.6726818

   Wankhade, M., Rao, A. C. S., & Kulkarni, C. (2022). A survey on sentiment analysis methods, applications, and challenges. Artificial Intelligence Review, 55(7), 5731–5780. https://doi.org/10.1007/s10462-022-10144-1






