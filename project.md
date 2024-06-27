zu unserem Ausfürlichen Bericht unseres Vorgehens: [Report](Report_Team_3.pdf)

1. Introduction
   
   Today, social media plays a central role in the exchange of opinions and information. Sentiment analysis is becoming increasingly important. Companies, political actors and researchers use this technology to capture and analyse public sentiment and opinions in real time. 
   Our motivation in this project is to make this possible and to gain experience with neural networks.
   Against this background, the research question is: How can machine learning methods be used effectively to classify sentiment using sentiment analysis in social media?



  The document is structured as follows:
  1. Introduction
  2. Related Work
  3. Methodology
  4. Results
  5. Discussion
  6. Conclusion
  7. bibliography
   
2. Related Work   
   
   There are three categories of machine learning: supervised learning, unsupervised learning and reinforcement learning. In supervised learning, we know the input and output, whereby the aim is to obtain the desired output based on the input. In contrast, the inputs and outputs are unknown in unsupervised learning. In reinforcement learning, information is collected through interaction with the environment, without a labelled data set (Jain et al., 2021).

   Sentiment analysis can be used, among other things, to identify opinions and sentiment about a product. Yadav et al compare different classifiers in order to assign tweets regarding certain products to the respective sentiment. This helps companies to plan the future of a product accordingly (Yadav et al., 2020).
   
   Wankhade et al presented the challenges of sentiment analysis. Dealing with ambiguous situations and irony is a central difficulty in sentiment analysis. It is a challenge for a machine to understand cultural allusions. In addition, performance in sentiment analysis can be influenced by grammatical or spelling errors in the text (Wankhade et al., 2022).


3. Methodology
   
   3.1. General Methodology 
   
      To achieve our project goals, we followed a structured and collaborative research process. We began by defining specific, measurable goals for each phase of the project, such as implementing the neural network, importing and preprocessing datasets, and improving model accuracy. Each objective served as an intermediate milestone guiding our progress. 
      
      Once objectives were set, team members independently gathered relevant external materials, including instructional YouTube videos, research papers, and relevant documentation. This equipped everyone with a solid foundation of knowledge related to the task at hand. After the initial research phase, we convened to share our findings, compiling information and addressing comprehension issues collectively. This collaborative knowledge-sharing ensured everyone was on the same page. 

      With a unified understanding, we transitioned to practical implementation, applying the gathered knowledge by coding in Python. This involved implementing the neural network, adding a language modell to interpret the given sentences, and fine-tuning the model to achieve higher accuracy. Throughout the implementation phase, we continuously evaluated our progress against the set objectives, making necessary adjustments and refinements based on feedback and results. For example, we decided to change the type of dataset we were using. We switched from a binary classification of hate speech to classifying different emotions in a sentence. The reason for this change was the relatively better performance of the neural network when it had more categories to distinguish between (6 instead of 2).

   3.2. Data Understanding and Preparation
   
      In our use case, we aim to develop a classification algorithm capable of categorizing text-based data into various emotional categories. To achieve this, we require datasets that contain specific labels to facilitate the training and evaluation of our model. Our selected dataset consists of 120,000 rows and three columns: ['tweet_id', 'sentiment', 'content']. The data types are int64 for 'tweet_id' and object for 'sentiment' and 'content'. The 'content' column contains the tweets, while the 'sentiment' column holds the labels, which include categories such as empty, sadness, enthusiasm, neutral, worry, surprise, love, fun, hate, happiness, boredom, relief, and anger.

      As outlined, we focus on classifying five primary sentiment categories. To ensure a comprehensive data foundation, we selected the five most frequently represented labels and included the "hate" category due to its significant representation of emotion in tweets. Our analysis revealed an uneven distribution of data in the dataset. To ensure a more balanced distribution for model training, we limited the number of data points per sentiment to the level of the "sadness" category. This approach resulted in the following distribution:

      Although the "love" and "hate" categories remain underrepresented, we consciously opted for this trade-off to maintain a sufficient data volume for model training. This strategy helps optimize the balance between data diversity and volume, enhancing the model's robustness.

   
   3.3. Modeling and Evaluation 
   
      For our project, we decided to create and train a neural network from scratch rather than using a pre-trained model. Our model architecture consists of several custom-programmed layers and neurons. Specifically, the network features 384 input neurons, representing the vector for the input sentence, and 6 output neurons, each corresponding to a different emotion category. To convert the input sentences into vectors, we implemented BERT, a language model that transforms sentences into corresponding vectors that can be processed by the neural network. Different sentences correspond to different vectors based on their meanings.

      To train the model, we initialized the weights for the connections between the layers with random values. During training, we improved these weights through multiple iterations. The process involves matrix multiplication to pass the input vector through the various layers, resulting in an emotion classification. We then compared the network's output to the pre-labeled emotion, calculated an error value, and used backpropagation to adjust the weights based on this error. This process was repeated multiple times with the same dataset to optimize the network's performance.

      For evaluation, we used a test dataset different from the training dataset to run the trained neural network and assess its accuracy. This final step ensured that our model generalized well to new, unseen data, allowing us to gauge its effectiveness in classifying emotions accurately.

5. Results
   - We have trained a neural network model specifically designed for tweet sentiment analysis. This model is capable of classifying the sentiment of tweets into categories.
   - A labeled dataset of tweets was used for training the model. This dataset includes various tweets along with their corresponding sentiment labels.
   - A user-friendly application where users can input text. The app processes this input through the trained model to predict the sentiment of the tweet.
   - An API endpoint that takes text input and returns the sentiment prediction. This API is integrated with the app to provide real-time sentiment analysis.

### Libraries and Tools Used

1. Python
2. Python librarys such as Pandas and Numy
3. A BERT Transformer Model
4.Streamlit for App Development

### Concept of the App

The app is designed to provide users with a simple and intuitive way to analyze the sentiment of tweets. Users can enter a tweet into a text box, the app sends the text to a server. The server processes the text through the trained neural network model and returns the predicted sentiment. This result is then displayed to the user on the app interface.

### Results on Unseen Data

By applying the trained models to unseen data (tweets that were not part of the training dataset), the following results have been observed:

1. **Accuracy:**
   - The model demonstrates a relativly low level of accuracy in predicting the sentiment of new tweets, effectively distinguishing between five Chategorys.

2. **Real-time Performance:**
   - The app provides real-time sentiment analysis, with predictions generated and displayed almost instantaneously after the user submits a tweet for analysis.

   
5. Discussion

   In our project we have developed an application capable of categorizing tweets to predict if people will react on the text with sadness, neutral, worried or if they hate it and so on. Our model was trained with on a dataset consisting of 120.000 tweets. 
Despite the comprehensiveness of the dataset, it´s important to acknowledge that some relevant tweets may have been missed and initially, we faced limitations in terms of training resources and GPU availability in Colab.
One of the challenges we encountered was the potential for inadvertent discrimination introduced by the algorithm. Biased language within the dataset could lead to inaccurate predictions. Therefore we took great care in selecting a dataset that was diverse and comprehensive, and had various examples of tweets to mitigate this issue.
Also we had to deal with limitations, which means that the accuracy is not as high as expected. Unfortunately it´s not possible to train such a model perfectly due to the limited memory.


   
8. Conclusion

   In conclusion, our project aimed to develop an application for sentiment analysis of tweets and this will be achieved through accurate data processing. 
Moving forward, our application holds significant potential to assist users in understanding how their tweets will be received by others, thanks to high-quality predictions generated by our model. By prioritizing accuracy, our project contributes to enhancing the overall quality and utility of sentiment analysis tools in the realm of social media. But we must not forget, that there is a long way to go before our application will be able to make predictions with accuracy and without making mistakes anymore.

10. bibliography


   Jain, P. K., Pamula, R., & Srivastava, G. (2021). A systematic literature review on machine learning applications for consumer sentiment analysis using online reviews. Computer Science Review, 41, 100413. https://doi.org/10.1016/j.cosrev.2021.100413

   Wankhade, M., Rao, A. C. S., & Kulkarni, C. (2022). A survey on sentiment analysis methods, applications, and challenges. Artificial Intelligence Review, 55(7), 5731–5780. https://doi.org/10.1007/s10462-022-10144-1
   
   Yadav, N., Kudale, O., Gupta, S., Rao, A., & Shitole, A. (2020). Twitter Sentiment Analysis Using Machine Learning For Product Evaluation. 2020 International Conference on Inventive Computation Technologies (ICICT), 181–185. https://doi.org/10.1109/ICICT48043.2020.9112381






